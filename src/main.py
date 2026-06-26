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

from pipeline import ExperimentPipeline
from utils.loaders import load_config, load_dataset, load_method, load_metric
from utils.visualizers import visualize_registration

logger = logging.getLogger(__name__)


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
    method = load_method(method_cfg["name"], **method_cfg["params"])

    dataset_cfg = config["dataset"]
    dataset = load_dataset(dataset_cfg["name"], **dataset_cfg["params"])

    metrics = [
        load_metric(metric_cfg["name"], **metric_cfg["params"])
        for metric_cfg in config["metrics"]
    ]

    return ExperimentPipeline(method, dataset, metrics, config)


def _parse_indices(value: str) -> int | list[int] | Literal["all"]:
    """Convert the raw `--index` value into the form `ExperimentPipeline.inference` expects.

    Parameters
    ----------
    value : str
        Raw `--index` option value: either the literal "all" or a
        comma-separated list of integers (e.g. "0,1,2").

    Returns
    -------
    int | list[int] | Literal["all"]
        "all" if `value` is "all", a single int if exactly one index was
        given, or a list of ints otherwise.
    """
    if value == "all":
        return "all"

    indices = [int(item) for item in value.split(",")]
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
    config = load_config(config_path)
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
    default="all",
    help="Comma-separated sample indices to run inference on (e.g. '0,1,2'), or 'all'.",
)
@click.option(
    "--save-transform",
    is_flag=True,
    default=False,
    help="Save the resulting transform(s) to config['transforms_path'].",
)
def inference(config_path: Path, index: str, save_transform: bool) -> None:
    """Run registration on the requested samples without computing metrics.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file describing the run.
    index : str
        Raw `--index` value, parsed via `_parse_indices`.
    save_transform : bool
        Whether to persist the resulting transform(s) to disk.
    """
    config = load_config(config_path)
    pipeline = build_pipeline(config)
    pipeline.inference(idx=_parse_indices(index), save_transform=save_transform)


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--index",
    type=int,
    required=True,
    help="Single sample index to run inference on and visualize.",
)
def visualize_single(config_path: Path, index: int) -> None:
    """Run inference on a single sample and visualize the registration result.

    The resulting transform is not saved to disk; it is only used to
    render the before/after registration via Open3D.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file describing the run.
    index : int
        Index of the dataset sample to visualize.
    """
    config = load_config(config_path)
    pipeline = build_pipeline(config)

    transform = pipeline.inference(idx=index, save_transform=False)
    sample = pipeline.dataset[index]

    visualize_registration(
        sample["infra_pts"].numpy(),
        sample["vehicle_pts"].numpy(),
        sample["T_infra_to_vehicle"].numpy(),
        transform,
    )


if __name__ == "__main__":
    cli()
