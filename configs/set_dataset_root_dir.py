"""Recursively set dataset.params.root_dir in configs.

Replaces the value of the ``root_dir`` entry inside the ``params`` mapping
nested under ``dataset`` in every ``.yaml`` file found (recursively) under a
given folder. Editing is done line-by-line rather than through a YAML dump so
comments and the original formatting are preserved.
"""

import logging
import re
from pathlib import Path

import click

logger = logging.getLogger(__name__)

NEW_ROOT_DIR = "/mnt/work/datasets/V2X-Real/V2X-Real-Lidar-128"

DATASET_RE = re.compile(r"^dataset:\s*$")
PARAMS_OPEN_RE = re.compile(r"^\s*params:\s*(#.*)?$")
TOP_LEVEL_KEY_RE = re.compile(r"^\S")
ROOT_DIR_RE = re.compile(r"^(\s*root_dir\s*:\s*)\S+(\s*)$")


def set_root_dir(lines: list[str]) -> list[str]:
    """Replace the dataset.params.root_dir value in place.

    Parameters
    ----------
    lines : list[str]
        Lines of the YAML file, including line endings.

    Returns
    -------
    list[str]
        Lines with the ``root_dir`` value under ``dataset.params`` replaced.
    """
    result: list[str] = []
    in_dataset = False
    in_params = False

    for line in lines:
        if DATASET_RE.match(line):
            in_dataset = True
            result.append(line)
            continue

        if in_dataset and not in_params and TOP_LEVEL_KEY_RE.match(line):
            in_dataset = False

        if in_dataset and PARAMS_OPEN_RE.match(line):
            in_params = True
            result.append(line)
            continue

        if in_params and ROOT_DIR_RE.match(line):
            match = ROOT_DIR_RE.match(line)
            result.append(f"{match.group(1)}{NEW_ROOT_DIR}{match.group(2)}")
            continue

        result.append(line)

    return result


@click.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(folder: Path) -> None:
    """Set dataset.params.root_dir in all YAML configs under FOLDER."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    yaml_files = sorted(folder.rglob("*.yaml"))
    if not yaml_files:
        logger.info("No .yaml files found under %s", folder)
        return

    for yaml_path in yaml_files:
        original_lines = yaml_path.read_text().splitlines(keepends=True)
        updated_lines = set_root_dir(original_lines)

        if updated_lines != original_lines:
            yaml_path.write_text("".join(updated_lines))
            logger.info("Updated %s", yaml_path)


if __name__ == "__main__":
    main()
