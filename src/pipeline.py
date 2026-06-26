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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from torch.utils.data import Dataset

from metrics import Metric
from registration import RegistrationMethod, Transform
from utils.helpers import compute_overlap_ratio

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """Evaluates a registration method against a dataset using a set of
    metrics, and persists the resulting metrics and configuration to disk.

    Each dataset item is expected to be a dict with keys `"source"` and
    `"target"` (point clouds passed to the registration method) and
    `"ground_truth"` (the :class:`~registration.base.Transform` that
    aligns `source` onto `target`).
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
            keys `"source"`, `"target"`, and `"ground_truth"`.
        metrics : list[Metric]
            Metrics to compute for each registered sample.
        config : dict[str, Any]
            Run configuration. Must contain an `"output_path"` key, the
            directory where results and the configuration are saved.
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
        registered_cloud, transform = self.method.register(
            sample["source"], sample["target"]
        )

        inputs = {
            "source": sample["source"],
            "registered_cloud": registered_cloud,
            "target": sample["target"],
            "transform": transform,
            "ground_truth": sample["ground_truth"],
        }

        results: dict[str, float] = {}
        for metric in self.metrics:
            results.update(metric.compute(**inputs))

        # Overlap is a property of the ground-truth alignment, not the
        # registration method, so it is computed once per sample here.
        results["overlap_ratio"] = compute_overlap_ratio(
            source=sample["source"],
            target=sample["target"],
            transformation=sample["ground_truth"].to_array(),
            threshold=self.config["overlap_threshold"],
        )
        return results

    def inference(
        self,
        idx: int | list[int] | Literal["all"] = "all",
        save_transform: bool = False,
    ) -> Transform | list[Transform]:
        """Run registration on the requested samples without computing metrics.

        Parameters
        ----------
        idx : int | list[int] | Literal["all"]
            A single sample index, a list of indices, or "all" for the
            whole dataset. Indices are processed in order.
        save_transform : bool
            If True, stack all resulting transforms into a single 3D
            array (via `Transform.to_array`) and save it with `np.save`
            to `self.config["transforms_path"]`, overwriting any
            existing file.

        Returns
        -------
        Transform | list[Transform]
            The estimated transform for each requested sample, in the
            same order as `idx`. A single `Transform` is returned when
            `idx` is a single int; a list is returned for "all" or a
            list of indices.
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
            _, transform = self.method.register(sample["source"], sample["target"])
            transforms.append(transform)

        if save_transform:
            matrices = np.stack([t.to_array() for t in transforms])
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
        """
        logger.info("Starting evaluation over %d samples", len(self.dataset))

        per_sample = {
            str(idx): self.evaluate_single(idx) for idx in range(len(self.dataset))
        }
        aggregate = self._aggregate(per_sample)

        #Since the path is sanitized and created on instantiation, this is not needed
        #self.output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "per_sample": per_sample,
            "aggregate": aggregate,
        }
        results_path = self.output_path / "metrics.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

        config_path = self.output_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(self.config, f)

        logger.info("Saved evaluation results to %s", results_path)

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
