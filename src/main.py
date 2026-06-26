"""Command-line interface for running registration experiments.

Each command reads a YAML configuration file, builds an
:class:`~pipeline.ExperimentPipeline` from it via `build_pipeline`, and
runs one of the pipeline's operations on it.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import click
import yaml

from pipeline import ExperimentPipeline
from utils.loaders import load_dataset, load_method, load_metric

logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read and parse a YAML configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        The parsed configuration.
    """
    with config_path.open() as f:
        return yaml.safe_load(f)


def _prepare_output_dir(config: dict[str, Any]) -> None:
    """Append a unique hex subfolder to `config["output_path"]` and create it.

    Mutates `config["output_path"]` in place so that the persisted run
    configuration reflects the actual output directory.

    Parameters
    ----------
    config : dict[str, Any]
        Run configuration. Must contain an `"output_path"` key.
    """
    base = Path(config["output_path"])

    # Collision is extremely unlikely but cheap to guard against.
    while (candidate := base / uuid.uuid4().hex).exists():
        continue

    candidate.mkdir(parents=True)
    config["output_path"] = str(candidate)
    logger.info("Created output directory %s", candidate)


def build_pipeline(config_path: Path) -> ExperimentPipeline:
    """Load a config file and instantiate the experiment pipeline it describes.

    Shared by every CLI command, since they all need the same method,
    dataset, and metrics setup before running a different operation.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    ExperimentPipeline
        The pipeline built from the configuration.
    """
    config = _load_config(config_path)
    _prepare_output_dir(config)

    method_cfg = config["registration_method"]
    method = load_method(method_cfg["name"], **method_cfg.get("params", {}))

    dataset_cfg = config["dataset"]
    dataset = load_dataset(dataset_cfg["name"], **dataset_cfg.get("params", {}))

    metrics = [
        load_metric(metric_cfg["name"], **metric_cfg.get("params", {}))
        for metric_cfg in config["metrics"]
    ]

    return ExperimentPipeline(method, dataset, metrics, config)


@click.group()
def cli() -> None:
    """Evaluate point cloud registration methods on lidar datasets."""


@cli.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def evaluate(config_path: Path) -> None:
    """Run a full evaluation of a registration method against a dataset.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file describing the run.
    """
    pipeline = build_pipeline(config_path)
    pipeline.evaluate()


if __name__ == "__main__":
    cli()