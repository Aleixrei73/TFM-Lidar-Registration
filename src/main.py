"""Command-line interface for running registration experiments.

Each command reads a YAML configuration file, builds an
:class:`~pipeline.ExperimentPipeline` from it via `build_pipeline`, and
runs one of the pipeline's operations on it.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Literal

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


def build_pipeline(config: dict[str, Any]) -> ExperimentPipeline:
    """Instantiate the experiment pipeline described by a config.

    Shared by every CLI command, since they all need the same method,
    dataset, and metrics setup before running a different operation.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed run configuration.

    Returns
    -------
    ExperimentPipeline
        The pipeline built from the configuration.
    """
    method_cfg = config["registration_method"]
    method = load_method(method_cfg["name"], **method_cfg.get("params", {}))

    dataset_cfg = config["dataset"]
    dataset = load_dataset(dataset_cfg["name"], **dataset_cfg.get("params", {}))

    metrics = [
        load_metric(metric_cfg["name"], **metric_cfg.get("params", {}))
        for metric_cfg in config["metrics"]
    ]

    return ExperimentPipeline(method, dataset, metrics, config)


def _parse_indices(values: tuple[str, ...]) -> int | list[int] | Literal["all"]:
    """Convert CLI `--index` values into the form `ExperimentPipeline.inference` expects.

    Parameters
    ----------
    values : tuple[str, ...]
        Raw `--index` option values. Each is either an integer or the
        literal "all".

    Returns
    -------
    int | list[int] | Literal["all"]
        "all" if no indices or "all" was given, a single int if exactly
        one index was given, or a list of ints otherwise.

    Raises
    ------
    click.BadParameter
        If "all" is combined with explicit indices.
    """
    if "all" in values:
        if len(values) > 1:
            raise click.BadParameter("'all' cannot be combined with explicit indices.")
        return "all"

    indices = [int(value) for value in values]
    return indices[0] if len(indices) == 1 else indices


@click.group()
def cli() -> None:
    """Evaluate point cloud registration methods on lidar datasets."""


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def evaluate(config_path: Path) -> None:
    """Run a full evaluation of a registration method against a dataset.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file describing the run.
    """
    config = _load_config(config_path)
    _prepare_output_dir(config)
    pipeline = build_pipeline(config)
    pipeline.evaluate()


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--index",
    "indices",
    multiple=True,
    default=("all",),
    help="Sample index to run inference on. May be repeated; defaults to 'all'.",
)
@click.option(
    "--save-transform",
    is_flag=True,
    default=False,
    help="Save the resulting transform(s) to config['transforms_path'].",
)
def inference(config_path: Path, indices: tuple[str, ...], save_transform: bool) -> None:
    """Run registration on the requested samples without computing metrics.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file describing the run.
    indices : tuple[str, ...]
        Raw `--index` values, parsed via `_parse_indices`.
    save_transform : bool
        Whether to persist the resulting transform(s) to disk.
    """
    config = _load_config(config_path)
    pipeline = build_pipeline(config)
    pipeline.inference(idx=_parse_indices(indices), save_transform=save_transform)


if __name__ == "__main__":
    cli()