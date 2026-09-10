"""Report missing combos in the VGICP voxel_resolution / max_correspondence_distance grid search.

Compares the expected grid, read from configs/vgicp-parametrized/grid-search-*-noise/,
against the runs found under an experiments folder. A combo is only considered
done if at least one matching run has metrics.json status "complete"; combos
whose only matching run(s) are "running" (or any other non-complete status)
are reported as missing, separately from combos with no matching run at all.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)

ROUND_NDIGITS = 2
NOISE_CONFIG_DIRS = {"low": "grid-search-low-noise", "high": "grid-search-high-noise"}
NOISE_PLANE_NAMES = {1: "low", 2: "high"}

ParamCombo = tuple[float, float]


def _load_yaml(path: Path) -> dict:
    """Load a YAML file as a dict."""
    with path.open() as f:
        return yaml.safe_load(f)


def _extract_params(config: dict) -> ParamCombo | None:
    """Extract (voxel_resolution, max_correspondence_distance) from a config dict.

    Parameters
    ----------
    config : dict
        Parsed config.yaml content.

    Returns
    -------
    ParamCombo | None
        Rounded (voxel_resolution, max_correspondence_distance) pair, or None
        if either field is missing/malformed.
    """
    try:
        params = config["registration_method"]["params"]
        return (
            round(float(params["voxel_resolution"]), ROUND_NDIGITS),
            round(float(params["max_correspondence_distance"]), ROUND_NDIGITS),
        )
    except (KeyError, TypeError):
        return None


def build_expected_grid(configs_root: Path) -> dict[str, dict[ParamCombo, Path]]:
    """Build the expected (voxel_resolution, max_correspondence_distance) grid per noise type.

    Parameters
    ----------
    configs_root : Path
        Path to configs/vgicp-parametrized, containing grid-search-low-noise
        and grid-search-high-noise subfolders.

    Returns
    -------
    dict[str, dict[ParamCombo, Path]]
        Maps "low"/"high" to a dict of param combo -> the config file that
        defines it, for every combo found in that noise type's config files.
    """
    expected: dict[str, dict[ParamCombo, Path]] = {}

    for noise_name, subdir in NOISE_CONFIG_DIRS.items():
        combos: dict[ParamCombo, Path] = {}
        for config_path in sorted((configs_root / subdir).rglob("*.yaml")):
            params = _extract_params(_load_yaml(config_path))
            if params is None:
                logger.warning("Skipping %s: missing swept parameters", config_path)
                continue
            combos[params] = config_path
        expected[noise_name] = combos

    return expected


def _read_status(metrics_path: Path) -> str:
    """Return the run status, or "unknown" if metrics.json is missing/malformed."""
    if not metrics_path.exists():
        return "unknown"
    try:
        with metrics_path.open() as f:
            return json.load(f).get("status", "unknown")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", metrics_path, exc)
        return "unknown"


def scan_experiments(experiments_root: Path) -> tuple[dict[str, set[ParamCombo]], dict[str, set[ParamCombo]]]:
    """Scan run folders, splitting param combos into completed vs. attempted-but-incomplete.

    Parameters
    ----------
    experiments_root : Path
        Path to the experiments folder, containing one subfolder per run.

    Returns
    -------
    tuple[dict[str, set[ParamCombo]], dict[str, set[ParamCombo]]]
        (completed, attempted_incomplete), each keyed by "low"/"high". A combo
        with duplicate runs lands in `completed` if any one of them succeeded.
    """
    completed: dict[str, set[ParamCombo]] = {"low": set(), "high": set()}
    attempted_incomplete: dict[str, set[ParamCombo]] = {"low": set(), "high": set()}

    for run_dir in sorted(p for p in experiments_root.iterdir() if p.is_dir()):
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            logger.warning("Skipping %s: no config.yaml", run_dir.name)
            continue

        config = _load_yaml(config_path)
        params = _extract_params(config)
        noise_plane = config.get("dataset", {}).get("params", {}).get("noise_plane")
        noise_name = NOISE_PLANE_NAMES.get(noise_plane)
        if params is None or noise_name is None:
            logger.warning("Skipping %s: malformed or unrecognized config.yaml", run_dir.name)
            continue

        status = _read_status(run_dir / "metrics.json")
        if status == "complete":
            completed[noise_name].add(params)
        else:
            attempted_incomplete[noise_name].add(params)

    return completed, attempted_incomplete


def _format_combo(combo: ParamCombo) -> str:
    voxel_resolution, max_correspondence_distance = combo
    return f"voxel_resolution={voxel_resolution}, max_correspondence_distance={max_correspondence_distance}"


def missing_config_paths(configs_root: Path, experiments_root: Path) -> list[Path]:
    """List the config files for every swept combo that is not yet complete.

    Parameters
    ----------
    configs_root : Path
        Path to configs/vgicp-parametrized.
    experiments_root : Path
        Path to the experiments folder to check completion against.

    Returns
    -------
    list[Path]
        Config file paths for missing combos, across both noise types,
        sorted for deterministic output.
    """
    expected = build_expected_grid(configs_root)
    completed, _ = scan_experiments(experiments_root)

    paths = [
        expected[noise_name][combo]
        for noise_name in ("low", "high")
        for combo in set(expected[noise_name]) - completed[noise_name]
    ]
    return sorted(paths)


def report_missing(configs_root: Path, experiments_root: Path) -> None:
    """Print, per noise type, the swept param combos that are not yet complete."""
    expected = build_expected_grid(configs_root)
    completed, attempted_incomplete = scan_experiments(experiments_root)

    for noise_name in ("low", "high"):
        missing = set(expected[noise_name]) - completed[noise_name]
        running = sorted(missing & attempted_incomplete[noise_name])
        not_started = sorted(missing - attempted_incomplete[noise_name])

        click.echo(f"\n=== {noise_name} noise ({len(missing)}/{len(expected[noise_name])} missing) ===")
        if running:
            click.echo(f"-- Running or incomplete, needs rerun ({len(running)}) --")
            for combo in running:
                click.echo(f"  {_format_combo(combo)}")

        if not_started:
            click.echo(f"-- Not started ({len(not_started)}) --")
            for combo in not_started:
                click.echo(f"  {_format_combo(combo)}")

        if not missing:
            click.echo("  All combos complete.")


@click.command()
@click.option(
    "--configs-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("configs/vgicp-parametrized"),
    show_default=True,
)
@click.option(
    "--experiments-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("experiments/vgicp-128-parametrized"),
    show_default=True,
)
@click.option(
    "--paths-only",
    is_flag=True,
    help="Print only the missing config file paths, one per line, for piping into a job submitter.",
)
def main(configs_root: Path, experiments_root: Path, paths_only: bool) -> None:
    """Report missing voxel_resolution / max_correspondence_distance combos per noise type."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if paths_only:
        for config_path in missing_config_paths(configs_root, experiments_root):
            click.echo(config_path)
    else:
        report_missing(configs_root, experiments_root)


if __name__ == "__main__":
    main()
