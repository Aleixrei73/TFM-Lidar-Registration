"""Experiment pipeline tying together a registration method, a dataset,
and a set of evaluation metrics.

This module defines :class:`ExperimentPipeline`, which runs a
:class:`~registration.base.RegistrationMethod` over a dataset of point
cloud pairs, computes a set of :class:`~metrics.base.Metric` values per
sample, and persists the results alongside the run configuration.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from metrics import Metric
from registration import RegistrationMethod
from utils.helpers import compute_overlap_ratio

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """Evaluates a registration method against a dataset using a set of
    metrics, and persists the resulting metrics and configuration to disk.

    Each dataset item is expected to follow the `V2XReal` schema: keys
    `"infra_pts"` and `"vehicle_pts"` (point clouds, as `torch.Tensor`)
    and `"T_infra_to_vehicle"` (a 4x4 `torch.Tensor` homogeneous transform
    that aligns `infra_pts` onto `vehicle_pts`).
    """

    def __init__(
        self,
        method: RegistrationMethod,
        dataset: Dataset,
        metrics: list[Metric],
        config: dict[str, Any],
    ) -> None:
        """Initialize the pipeline.

        Parameters
        ----------
        method : RegistrationMethod
            Registration method to evaluate.
        dataset : Dataset
            Dataset of point cloud pairs. Each item must be a dict with
            keys `"infra_pts"`, `"vehicle_pts"`, and
            `"T_infra_to_vehicle"`.
        metrics : list[Metric]
            Metrics to compute for each registered sample.
        config : dict[str, Any]
            Run configuration. Must contain an `"output_path"` key, the
            directory where results and the configuration are saved. May
            contain a `"checkpoint_every"` key (default 10) controlling
            how often `evaluate` checkpoints partial results to disk.
        """
        self.method = method
        self.dataset = dataset
        self.metrics = metrics
        self.config = config
        self.output_path = Path(config["output_path"])

    def evaluate_single(self, idx: int) -> dict[str, float]:
        """Register a single dataset sample and compute all metrics on it.

        Parameters
        ----------
        idx : int
            Index of the sample in `dataset`.

        Returns
        -------
        dict[str, float]
            Mapping from metric name to its computed value, merged across
            all metrics in `self.metrics`.
        """
        sample = self.dataset[idx]
        infra_pts = sample["infra_pts"].numpy()
        vehicle_pts = sample["vehicle_pts"].numpy()
        transform_gt = sample["T_infra_to_vehicle"].numpy()
        transform_imu = sample["T_infra_to_vehicle_imu"].numpy()

        registered_cloud, transform = self.method.register(
            infra_pts, vehicle_pts, transform_imu=transform_imu
        )

        inputs = {
            "infra_pts": infra_pts,
            "registered_cloud": registered_cloud,
            "vehicle_pts": vehicle_pts,
            "transform_pred": transform,
            "transform_gt": transform_gt,
        }

        results: dict[str, float] = {}
        for metric in self.metrics:
            results.update(metric.compute(**inputs))

        # Overlap is a property of the ground-truth alignment, not the
        # registration method, so it is computed once per sample here.
        results["overlap_ratio"] = compute_overlap_ratio(
            source=infra_pts,
            target=vehicle_pts,
            transformation=transform_gt,
            threshold=self.config["overlap_threshold"],
        )
        return results

    def inference(
        self,
        idx: int | list[int] | Literal["all"] = "all",
        save_transform: bool = False,
    ) -> np.ndarray | list[np.ndarray]:
        """Run registration on the requested samples without computing metrics.

        Parameters
        ----------
        idx : int | list[int] | Literal["all"]
            A single sample index, a list of indices, or "all" for the
            whole dataset. Indices are processed in order.
        save_transform : bool
            If True, stack all resulting 4x4 transforms into a single 3D
            array and save it with `np.save` to
            `self.config["transforms_path"]`, overwriting any existing
            file.

        Returns
        -------
        np.ndarray | list[np.ndarray]
            The estimated 4x4 transform for each requested sample, in the
            same order as `idx`. A single transform is returned when `idx`
            is a single int; a list is returned for "all" or a list of
            indices.
        """
        single = isinstance(idx, int)
        if idx == "all":
            indices = list(range(len(self.dataset)))
        elif single:
            indices = [idx]
        else:
            indices = list(idx)

        transforms = []
        for index in indices:
            sample = self.dataset[index]
            infra_pts = sample["infra_pts"].numpy()
            vehicle_pts = sample["vehicle_pts"].numpy()
            transform_imu = sample["T_infra_to_vehicle_imu"].numpy()
            _, transform = self.method.register(
                infra_pts, vehicle_pts, transform_imu=transform_imu
            )
            transforms.append(transform)

        if save_transform:
            matrices = np.stack(transforms)
            transforms_path = Path(self.config["transforms_path"])
            transforms_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(transforms_path, matrices)
            logger.info("Saved %d transforms to %s", len(transforms), transforms_path)

        return transforms[0] if single else transforms

    def evaluate(self) -> None:
        """Compute metrics over the full dataset and save the results.

        Per-sample and aggregate (mean, std) metric values are saved to
        `metrics.json`, and the run configuration is saved to
        `config.yaml`, both inside `self.output_path`. Existing files at
        those paths are overwritten.

        `metrics.json` is checkpointed to disk every
        `self.config["checkpoint_every"]` samples (default 10), so an
        interrupted run leaves behind a usable partial result with
        `"status": "running"` instead of losing all progress. The final
        write always happens regardless of the checkpoint interval, and
        is marked `"status": "complete"`.
        """
        logger.info("Starting evaluation over %d samples", len(self.dataset))
        checkpoint_every = self.config.get("checkpoint_every", 10)

        # Since the path is sanitized and created on instantiation, this is not needed
        # self.output_path.mkdir(parents=True, exist_ok=True)

        # Saved upfront, alongside the run, so it is available even if the
        # evaluation loop below is interrupted partway through.
        config_path = self.output_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(self.config, f)

        results_path = self.output_path / "metrics.json"
        per_sample = {}
        num_samples = len(self.dataset)
        for idx in tqdm(range(num_samples), desc="Evaluating samples"):
            per_sample[str(idx)] = self.evaluate_single(idx)

            is_last = idx == num_samples - 1
            if is_last or (idx + 1) % checkpoint_every == 0:
                self._write_results(
                    results_path,
                    per_sample,
                    status="complete" if is_last else "running",
                )

        logger.info("Saved evaluation results to %s", results_path)

    @staticmethod
    def _write_results(
        results_path: Path,
        per_sample: dict[str, dict[str, float]],
        status: Literal["running", "complete"],
    ) -> None:
        """Atomically write per-sample and aggregate results to disk.

        Writes to a temporary file in the same directory and then
        replaces `results_path`, so a crash mid-write never leaves a
        truncated or corrupt `metrics.json` behind.

        Parameters
        ----------
        results_path : Path
            Destination path for the results file.
        per_sample : dict[str, dict[str, float]]
            Metric values for every sample evaluated so far.
        status : Literal["running", "complete"]
            Whether this is a mid-run checkpoint or the final result.
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "per_sample": per_sample,
            "aggregate": ExperimentPipeline._aggregate(per_sample),
        }

        tmp_path = results_path.with_suffix(".json.tmp")
        with tmp_path.open("w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp_path, results_path)

    @staticmethod
    def _aggregate(
        per_sample: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Compute the mean and standard deviation of each metric across
        all samples.

        Parameters
        ----------
        per_sample : dict[str, dict[str, float]]
            Mapping from sample index (as a string) to its metric dict.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping from metric name to a `{"mean": ..., "std": ...}` dict.
        """
        metric_names = {name for sample in per_sample.values() for name in sample}

        aggregate: dict[str, dict[str, float]] = {}
        for name in metric_names:
            # Not every sample is guaranteed to report every metric name.
            values = np.array(
                [sample[name] for sample in per_sample.values() if name in sample]
            )
            aggregate[name] = {"mean": float(values.mean()), "std": float(values.std())}
        return aggregate
