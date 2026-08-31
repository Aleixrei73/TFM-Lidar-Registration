"""Recursively strip voxel_resolution/correspondance_distance from configs.

Removes the ``voxel_resolution`` and ``correspondance_distance`` entries
from the ``params`` flow-mapping nested under ``registration_method`` in
every ``.yaml`` file found (recursively) under a given folder. Editing is
done line-by-line rather than through a YAML dump so comments and the
original flow-style formatting are preserved.
"""

import logging
import re
from pathlib import Path

import click

logger = logging.getLogger(__name__)

KEYS_TO_REMOVE = ("voxel_resolution", "correspondance_distance")

REGISTRATION_METHOD_RE = re.compile(r"^registration_method:\s*$")
PARAMS_OPEN_RE = re.compile(r"^\s*params:\s*\{\s*$")
TOP_LEVEL_KEY_RE = re.compile(r"^\S")
REMOVE_KEY_RE = re.compile(r"^\s*(" + "|".join(KEYS_TO_REMOVE) + r")\s*:")


def strip_params_keys(lines: list[str]) -> list[str]:
    """Remove target keys from the registration_method.params block.

    Parameters
    ----------
    lines : list[str]
        Lines of the YAML file, including line endings.

    Returns
    -------
    list[str]
        Lines with the target keys removed and a dangling trailing comma
        (left behind if the removed key was the last entry) fixed up.
    """
    result: list[str] = []
    in_registration_method = False
    in_params = False

    for line in lines:
        if REGISTRATION_METHOD_RE.match(line):
            in_registration_method = True
            result.append(line)
            continue

        if in_registration_method and not in_params and TOP_LEVEL_KEY_RE.match(line):
            in_registration_method = False

        if in_registration_method and PARAMS_OPEN_RE.match(line):
            in_params = True
            result.append(line)
            continue

        if in_params:
            if REMOVE_KEY_RE.match(line):
                continue
            if line.strip().startswith("}"):
                in_params = False
                in_registration_method = False
                # Drop a dangling trailing comma left by a removed last entry.
                if result and result[-1].rstrip().endswith(","):
                    stripped = result[-1].rstrip()
                    ending = result[-1][len(stripped):]
                    result[-1] = stripped[:-1] + ending

        result.append(line)

    return result


@click.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(folder: Path) -> None:
    """Strip voxel_resolution/correspondance_distance from all YAML configs under FOLDER."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    yaml_files = sorted(folder.rglob("*.yaml"))
    if not yaml_files:
        logger.info("No .yaml files found under %s", folder)
        return

    for yaml_path in yaml_files:
        original_lines = yaml_path.read_text().splitlines(keepends=True)
        updated_lines = strip_params_keys(original_lines)

        if updated_lines != original_lines:
            yaml_path.write_text("".join(updated_lines))
            logger.info("Updated %s", yaml_path)


if __name__ == "__main__":
    main()
