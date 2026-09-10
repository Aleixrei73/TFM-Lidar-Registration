"""Generate a VGICP grid search of config files over voxel_resolution and max_correspondence_distance.

For each voxel_resolution value in the sweep, a folder is created containing
one config per max_correspondence_distance value. All other fields are
copied verbatim from the base config.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

VOXEL_RESOLUTION_PATTERN = re.compile(r"(voxel_resolution\s*:\s*)([0-9.]+)")
MAX_CORRESPONDENCE_DISTANCE_PATTERN = re.compile(r"(max_correspondence_distance\s*:\s*)([0-9.]+)")


def _grid_values(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive, rounded arithmetic sequence from `start` to `stop`.

    Parameters
    ----------
    start : float
        First value of the sequence.
    stop : float
        Last value of the sequence (inclusive).
    step : float
        Spacing between consecutive values.

    Returns
    -------
    list[float]
        Values rounded to one decimal place to avoid floating-point drift
        (e.g. 0.2 * 3 == 0.6000000000000001).
    """
    count = round((stop - start) / step) + 1
    return [round(start + i * step, 1) for i in range(count)]


def _format_value(value: float) -> str:
    """Format a grid value for insertion into a YAML file, e.g. 1.0 -> '1.0'."""
    return f"{value:.1f}"


def _format_slug(value: float) -> str:
    """Format a grid value for use in folder/file names, e.g. 0.2 -> '0-2'."""
    return _format_value(value).replace(".", "-")


def generate_grid_search(
    base_config_path: Path,
    output_root: Path,
    voxel_resolutions: list[float],
    max_correspondence_distances: list[float],
) -> None:
    """Write one config per (voxel_resolution, max_correspondence_distance) pair.

    Parameters
    ----------
    base_config_path : Path
        Template YAML config. Its text is reused verbatim except for the
        two swept fields, so formatting and comments are preserved.
    output_root : Path
        Folder under which one subfolder per voxel_resolution is created.
    voxel_resolutions : list[float]
        Grid values for `registration_method.params.voxel_resolution`.
    max_correspondence_distances : list[float]
        Grid values for `registration_method.params.max_correspondence_distance`.
    """
    base_text = base_config_path.read_text()

    for voxel_resolution in voxel_resolutions:
        folder = output_root / f"voxel-resolution-{_format_slug(voxel_resolution)}"
        folder.mkdir(parents=True, exist_ok=True)
        text_with_voxel_resolution = VOXEL_RESOLUTION_PATTERN.sub(
            rf"\g<1>{_format_value(voxel_resolution)}", base_text
        )

        for max_correspondence_distance in max_correspondence_distances:
            text = MAX_CORRESPONDENCE_DISTANCE_PATTERN.sub(
                rf"\g<1>{_format_value(max_correspondence_distance)}", text_with_voxel_resolution
            )

            config_path = folder / f"vgicp-parametrized-mcd-{_format_slug(max_correspondence_distance)}.yaml"
            config_path.write_text(text)
            logger.info("Wrote %s", config_path)


@click.command()
@click.option(
    "--base-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/vgicp-parametrized/vgicp-parametrized.yaml"),
    show_default=True,
)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("configs/vgicp-parametrized/grid-search"),
    show_default=True,
)
@click.option("--start", type=float, default=0.2, show_default=True)
@click.option("--stop", type=float, default=2.0, show_default=True)
@click.option("--step", type=float, default=0.2, show_default=True)
def main(base_config: Path, output_root: Path, start: float, stop: float, step: float) -> None:
    """Generate the voxel_resolution / max_correspondence_distance grid search."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    values = _grid_values(start, stop, step)
    generate_grid_search(base_config, output_root, values, values)


if __name__ == "__main__":
    main()
