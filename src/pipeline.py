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
from collections import deque
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
            how often `evaluate` checkpoints partial results to disk. May
            contain a `"use_previous"` key (default False); if True, the
            initial transform guess for a sample is the previous sample's
            registered transform instead of its own IMU-derived transform,
            except on samples with `meta["is_first_frame"]` True, which
            always use their own IMU-derived transform.
            May contain an `"overlap"` key (default 0.0), a float in
            [0, 1]. When set together with `"use_previous"`, `evaluate`
            and `evaluate_single` skip any sample whose overlap ratio
            does not exceed this threshold (reporting only its overlap
            ratio), and the first sample exceeding it after such a gap
            is treated as a first frame: its own IMU-derived transform
            is used as the initial guess regardless of
            `meta["is_first_frame"]`, and its metrics are marked with
            `"first_frame": True`. In this mode, `meta["is_first_frame"]`
            is instead used only to detect a scene change, which is
            accounted for the same way (own IMU-derived transform and
            `"first_frame": True`) instead of reusing the previous
            sample's registered transform.
            May contain an `"order"` key (default 0), either 0 or 1. If
            1, once two consecutive registered transforms are available
            for the current first-frame chain, the initial guess that
            would otherwise be the previous sample's registered
            transform is instead an order-1 (constant-velocity)
            extrapolation from the previous two registered transforms.
            The chain of two transforms resets whenever a new first
            frame is reached.
        """
        self.method = method
        self.dataset = dataset
        self.metrics = metrics
        self.config = config
        self.output_path = Path(config["output_path"])
        self.use_previous = config.get("use_previous", False)
        self.overlap = config.get("overlap", 0.0)
        self.order = config.get("order", 0)
        self._previous_transform: np.ndarray | None = None
        self._transform_queue: deque[np.ndarray] = deque(maxlen=2)
        self._awaiting_first_frame: bool = True

    def _resolve_transform_imu(self, sample: dict) -> np.ndarray:
        """Get the initial transform guess for a sample.

        Used by `inference`, and as the fallback of
        `_resolve_transform_imu_for_evaluation` when `self.use_previous`
        is False.

        Parameters
        ----------
        sample : dict
            Dataset item, as returned by `self.dataset[idx]`.

        Returns
        -------
        np.ndarray
            When `self.use_previous` is True and the sample is not a
            first frame: an order-1 extrapolation from the previous two
            registered transforms if `self.order` is 1 and both are
            available, otherwise `self._previous_transform`. The
            sample's own IMU-derived transform otherwise, which also
            resets the order-1 queue.
        """
        if self.use_previous and not sample["meta"]["is_first_frame"]:
            if self.order == 1 and len(self._transform_queue) == 2:
                return self._extrapolate_transform(self._transform_queue)
            return self._previous_transform

        if self.order == 1:
            self._transform_queue.clear()
        return sample["T_infra_to_vehicle_imu"].numpy()

    @staticmethod
    def _extrapolate_transform(queue: deque[np.ndarray]) -> np.ndarray:
        """Order-1 (constant-velocity) extrapolation from the previous
        two registered transforms.

        Parameters
        ----------
        queue : deque[np.ndarray]
            The previous two registered 4x4 transforms, oldest first.

        Returns
        -------
        np.ndarray
            The predicted 4x4 transform, continuing the translation delta
            between the two queued transforms by one more step and
            keeping the most recent rotation unchanged.
        """
        transform_prev2, transform_prev1 = queue
        translation_prev2 = transform_prev2[:3, 3]
        rotation_prev1, translation_prev1 = (
            transform_prev1[:3, :3],
            transform_prev1[:3, 3],
        )

        delta_translation = translation_prev1 - translation_prev2

        predicted = np.eye(4)
        predicted[:3, :3] = rotation_prev1
        predicted[:3, 3] = translation_prev1 + delta_translation
        return predicted

    def _resolve_transform_imu_for_evaluation(
        self, sample: dict
    ) -> tuple[np.ndarray, bool]:
        """Get the initial transform guess for a sample during `evaluate`,
        accounting for overlap-based first-frame selection and order-1
        extrapolation.

        Only used when `self.use_previous` is True; a sample becomes a
        first frame either because it is the first to exceed
        `self.overlap` after a gap of skipped samples, or because
        `meta["is_first_frame"]` marks a scene change. Either case is
        accounted for the same way: the sample's own IMU-derived
        transform is used as the initial guess, disregarding the
        previous sample's registered transform, and the order-1 queue
        is reset since it no longer applies to the new chain.

        When continuing a chain and `self.order` is 1, the initial
        guess is instead an order-1 extrapolation from the previous two
        registered transforms, once both are available.

        Parameters
        ----------
        sample : dict
            Dataset item, as returned by `self.dataset[idx]`.

        Returns
        -------
        tuple[np.ndarray, bool]
            The initial transform guess, and whether this sample is a
            first frame.
        """
        if not self.use_previous:
            return self._resolve_transform_imu(sample), False

        if self._awaiting_first_frame or sample["meta"]["is_first_frame"]:
            self._awaiting_first_frame = False
            if self.order == 1:
                self._transform_queue.clear()
            return sample["T_infra_to_vehicle_imu"].numpy(), True

        if self.order == 1 and len(self._transform_queue) == 2:
            return self._extrapolate_transform(self._transform_queue), False

        return self._previous_transform, False

    def evaluate_single(self, idx: int) -> dict[str, float]:
        """Register a single dataset sample and compute all metrics on it.

        When `self.use_previous` is True and `self.overlap` is set, a
        sample whose overlap ratio does not exceed `self.overlap` is
        skipped: registration and the other metrics are not computed,
        and only `"overlap_ratio"` is reported. The next sample to
        exceed the threshold becomes a first frame, reported with
        `"first_frame": True` (see `_resolve_transform_imu_for_evaluation`).

        Parameters
        ----------
        idx : int
            Index of the sample in `dataset`.

        Returns
        -------
        dict[str, float]
            Mapping from metric name to its computed value, merged across
            all metrics in `self.metrics`. Just `{"overlap_ratio": ...}`
            for a skipped sample.
        """
        sample = self.dataset[idx]
        infra_pts = sample["infra_pts"].numpy()
        vehicle_pts = sample["vehicle_pts"].numpy()
        transform_gt = sample["T_infra_to_vehicle"].numpy()

        # Overlap is a property of the ground-truth alignment, not the
        # registration method, so it is computed once per sample here.
        overlap_ratio = compute_overlap_ratio(
            source=infra_pts,
            target=vehicle_pts,
            transformation=transform_gt,
            threshold=self.config["overlap_threshold"],
        )

        if self.use_previous and overlap_ratio <= self.overlap:
            self._awaiting_first_frame = True
            return {"overlap_ratio": overlap_ratio}

        transform_imu, is_first_frame = self._resolve_transform_imu_for_evaluation(
            sample
        )

        registered_cloud, transform = self.method.register(
            infra_pts, vehicle_pts, transform_imu=transform_imu
        )
        self._previous_transform = transform
        if self.order == 1:
            self._transform_queue.append(transform)

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

        results["overlap_ratio"] = overlap_ratio
        if is_first_frame:
            results["first_frame"] = True

        return results

    def inference(
        self,
        idx: int | list[int] | Literal["all"] = "all",
        save_transform: bool = False,
    ) -> np.ndarray | list[np.ndarray]:
        """Run registration on the requested samples without computing metrics.

        Follows `self.use_previous` and `self.order` the same way as
        `evaluate` (see `_resolve_transform_imu`), using scene changes
        (`meta["is_first_frame"]`) as the only first-frame signal, since
        there is no overlap-based skipping here.

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
        self._previous_transform = None
        self._transform_queue.clear()

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
            transform_imu = self._resolve_transform_imu(sample)
            _, transform = self.method.register(
                infra_pts, vehicle_pts, transform_imu=transform_imu
            )
            self._previous_transform = transform
            if self.order == 1:
                self._transform_queue.append(transform)
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
        self._previous_transform = None
        self._transform_queue.clear()
        self._awaiting_first_frame = True

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

        Samples skipped for insufficient overlap (reporting only
        `"overlap_ratio"`) are excluded from aggregation entirely, and
        `"first_frame"` is never aggregated.

        Parameters
        ----------
        per_sample : dict[str, dict[str, float]]
            Mapping from sample index (as a string) to its metric dict.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping from metric name to a `{"mean": ..., "std": ...}` dict.
        """
        registered_samples = [
            sample
            for sample in per_sample.values()
            if sample.keys() != {"overlap_ratio"}
        ]
        metric_names = {
            name
            for sample in registered_samples
            for name in sample
            if name != "first_frame"
        }

        aggregate: dict[str, dict[str, float]] = {}
        for name in metric_names:
            # Not every sample is guaranteed to report every metric name.
            values = np.array(
                [sample[name] for sample in registered_samples if name in sample]
            )
            aggregate[name] = {"mean": float(values.mean()), "std": float(values.std())}
        return aggregate
