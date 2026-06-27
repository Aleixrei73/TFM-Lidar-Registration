"""Summarize per-run `metrics.json` files into parquet tables.

Walks `experiments/<algorithm>/<uuid>/metrics.json`, and writes one parquet
file with the aggregate metrics already stored in each run, plus three more
where the aggregate is recomputed restricted to samples whose overlap ratio
exceeds 0.3, 0.5 and 0.8 respectively.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OVERLAP_THRESHOLDS = (0.3, 0.5, 0.8)


def find_metrics_files(experiments_dir: Path) -> list[Path]:
    """Find every `metrics.json` directly two levels below `experiments_dir`.

    Parameters
    ----------
    experiments_dir : Path
        Root directory containing one subfolder per algorithm, each with
        one subfolder per run (named by uuid).

    Returns
    -------
    list[Path]
        Paths to valid `metrics.json` files, skipping any that aren't at
        the expected `<algorithm>/<uuid>/metrics.json` depth.
    """
    found = []
    for path in experiments_dir.rglob("metrics.json"):
        if len(path.relative_to(experiments_dir).parts) != 3:
            logger.warning("Skipping %s: unexpected path depth", path)
            continue
        found.append(path)
    return found


def aggregate_metrics(samples: list[dict[str, float]], metric_names: set[str]) -> dict[str, dict[str, float]]:
    """Compute the mean and standard deviation of each metric across samples.

    Mirrors `Pipeline._aggregate` (src/pipeline.py): unions metric names
    across samples and ignores samples missing a given metric.

    Parameters
    ----------
    samples : list[dict[str, float]]
        Per-sample metric dicts to aggregate over.
    metric_names : set[str]
        Metric names to compute aggregates for, used even when `samples`
        is empty so that the result always has the same keys.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from metric name to a `{"mean": ..., "std": ...}` dict.
        Both are `NaN` for a metric with no contributing samples.
    """
    aggregate = {}
    for name in metric_names:
        values = np.array([sample[name] for sample in samples if name in sample])
        if values.size == 0:
            aggregate[name] = {"mean": float("nan"), "std": float("nan")}
        else:
            aggregate[name] = {"mean": float(values.mean()), "std": float(values.std())}
    return aggregate


def build_row(algorithm: str, run_id: str, timestamp: str, n_samples: int, aggregate: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Flatten run identity and an aggregate dict into a single parquet row.

    Parameters
    ----------
    algorithm : str
        Name of the algorithm's experiments subfolder.
    run_id : str
        Uuid of the run.
    timestamp : str
        Timestamp recorded in the run's `metrics.json`.
    n_samples : int
        Number of samples the aggregate was computed over.
    aggregate : dict[str, dict[str, float]]
        Mapping from metric name to its `{"mean", "std"}` dict.

    Returns
    -------
    dict[str, Any]
        Row with `{metric}_mean` / `{metric}_std` columns alongside the
        run's identifying fields.
    """
    row = {
        "algorithm": algorithm,
        "run_id": run_id,
        "timestamp": timestamp,
        "n_samples": n_samples,
    }
    for metric, stats in aggregate.items():
        row[f"{metric}_mean"] = stats["mean"]
        row[f"{metric}_std"] = stats["std"]
    return row


def process_metrics_file(path: Path) -> tuple[dict[str, Any], dict[float, dict[str, Any]]] | None:
    """Build the full-data row and per-threshold rows for one `metrics.json`.

    Parameters
    ----------
    path : Path
        Path to the `metrics.json` file, expected at
        `experiments/<algorithm>/<uuid>/metrics.json`.

    Returns
    -------
    tuple[dict[str, Any], dict[float, dict[str, Any]]] | None
        The full-data row and a mapping from overlap threshold to its
        recomputed row, or `None` if the file was malformed.
    """
    algorithm, run_id = path.parts[-3], path.parts[-2]

    try:
        with path.open() as f:
            data = json.load(f)
        timestamp = data["timestamp"]
        per_sample = data["per_sample"]
        aggregate = data["aggregate"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Skipping %s: malformed metrics.json (%s)", path, exc)
        return None

    full_row = build_row(algorithm, run_id, timestamp, len(per_sample), aggregate)

    metric_names = set(aggregate)
    samples = list(per_sample.values())
    threshold_rows = {}
    for threshold in OVERLAP_THRESHOLDS:
        filtered = [s for s in samples if s.get("overlap_ratio", float("-inf")) >= threshold]
        threshold_aggregate = aggregate_metrics(filtered, metric_names)
        threshold_rows[threshold] = build_row(algorithm, run_id, timestamp, len(filtered), threshold_aggregate)

    return full_row, threshold_rows


def main() -> None:
    """Summarize all experiment runs into parquet files under `experiments/`."""
    logging.basicConfig(level=logging.INFO)
    experiments_dir = Path(__file__).parent

    metrics_files = find_metrics_files(experiments_dir)
    logger.info("Found %d metrics.json files", len(metrics_files))

    full_rows = []
    threshold_row_lists: dict[float, list[dict[str, Any]]] = {t: [] for t in OVERLAP_THRESHOLDS}
    skipped = 0
    for path in metrics_files:
        result = process_metrics_file(path)
        if result is None:
            skipped += 1
            continue
        full_row, threshold_rows = result
        full_rows.append(full_row)
        for threshold, row in threshold_rows.items():
            threshold_row_lists[threshold].append(row)

    logger.info("Processed %d runs, skipped %d malformed files", len(full_rows), skipped)

    pd.DataFrame(full_rows).to_parquet(experiments_dir / "summaries/summary.parquet", index=False)
    for threshold, rows in threshold_row_lists.items():
        out_path = experiments_dir / f"summaries/summary_overlap_{threshold}.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
