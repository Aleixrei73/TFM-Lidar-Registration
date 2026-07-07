"""
Beamagine Dataset for Point Cloud Registration
===============================================
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from torch.utils.data import Dataset

MIN_SCENE_NUMBER = 1
MAX_SCENE_NUMBER = 7

# Fields kept from a sample_data entry once it is matched into a pair.
SAMPLE_FIELDS = ("filename", "calibrated_sensor_token", "timestamp", "sample_token")


def load_json(path: Path) -> list:
    """Load a JSON file into a list of dicts."""
    with open(path, "r") as f:
        return json.load(f)


def _project_sample(sample: dict) -> dict:
    """Reduce a sample_data entry to the fields needed downstream."""
    return {field: sample[field] for field in SAMPLE_FIELDS}


class Beamagine(Dataset):
    """
    PyTorch Dataset for point cloud registration on Beamagine.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root folder.
    json_folder : str
        Name of the subfolder of ``root_dir`` containing the JSON files
        to be read.
    scene_number : int
        Scene identifier, between 1 and 7 (inclusive).
    """

    def __init__(self, root_dir: str, json_folder: str, scene_number: int) -> None:
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        if not root_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {root_dir}")

        json_dir = root_dir / json_folder
        if not json_dir.exists():
            raise FileNotFoundError(f"Directory not found: {json_dir}")
        if not json_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {json_dir}")

        if not MIN_SCENE_NUMBER <= scene_number <= MAX_SCENE_NUMBER:
            raise ValueError(
                f"scene_number must be between {MIN_SCENE_NUMBER} and "
                f"{MAX_SCENE_NUMBER}, got {scene_number}"
            )

        self.root_dir = root_dir
        self.json_dir = json_dir
        self.scene_number = scene_number
        self.sample_pairs = self._build_sample_correspondances()

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def _build_scene_correspondances(self) -> Tuple[dict, dict]:
        """
        Find the LiDAR sample_data entries for this scene's agents.

        Returns
        -------
        tuple of (dict, dict)
            ``(car_sample, infra_sample)``, the ``sample_data`` entries
            holding the first LiDAR (``.bin``) reading of the car and
            static (infrastructure) agents for ``self.scene_number``.
        """
        scenes = load_json(self.json_dir / "scene.json")

        car_name = f"car_0{self.scene_number}"
        infra_name = f"static_0{self.scene_number}"
        car_scene = next((s for s in scenes if s["name"] == car_name), None)
        infra_scene = next((s for s in scenes if s["name"] == infra_name), None)
        if car_scene is None or infra_scene is None:
            raise ValueError(
                f"Could not find both '{car_name}' and '{infra_name}' in scene.json"
            )

        sample_data = load_json(self.json_dir / "sample_data.json")

        car_sample = self._find_lidar_sample(
            sample_data, car_scene["first_sample_token"]
        )
        infra_sample = self._find_lidar_sample(
            sample_data, infra_scene["first_sample_token"]
        )

        return car_sample, infra_sample

    @staticmethod
    def _find_lidar_sample(sample_data: list, sample_token: str) -> dict:
        """
        Find the unique LiDAR entry for a sample token.

        A ``sample_token`` is shared by multiple ``sample_data`` entries,
        one per sensor modality, so the LiDAR reading is disambiguated
        by filtering on ``fileformat == "bin"``.
        """
        matches = [
            d
            for d in sample_data
            if d["sample_token"] == sample_token and d["fileformat"] == "bin"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one 'bin' sample_data entry for token "
                f"'{sample_token}', found {len(matches)}"
            )

        return matches[0]

    def _build_sample_correspondances(self) -> List[Tuple[dict, dict]]:
        """
        Pair up car and infrastructure LiDAR samples by closest timestamp.

        Starting from the first car/infrastructure samples, the stream
        that lags behind is advanced sample by sample (via the ``next``
        token) until it overtakes the other stream's timestamp; whichever
        of the two candidates straddling that crossing is closest in time
        is kept as the match. The lead stream then advances one sample,
        swapping which stream leads if the advance puts it behind again,
        and the process repeats until either stream runs out of samples.

        Returns
        -------
        list of (dict, dict)
            Ordered ``(car_sample, infra_sample)`` pairs, each reduced to
            ``filename``, ``calibrated_sensor_token``, ``timestamp`` and
            ``sample_token``. Also stored as ``self.sample_pairs``.
        """
        car_sample, infra_sample = self._build_scene_correspondances()
        by_token = {
            d["token"]: d
            for d in load_json(self.json_dir / "sample_data.json")
        }

        if car_sample["timestamp"] >= infra_sample["timestamp"]:
            fixed, behind, fixed_is_car = car_sample, infra_sample, True
        else:
            fixed, behind, fixed_is_car = infra_sample, car_sample, False

        pairs: List[Tuple[dict, dict]] = []

        while True:
            next_behind = self._next_lidar_sample(behind, by_token)
            if next_behind is None:
                break

            diff_behind = fixed["timestamp"] - behind["timestamp"]
            diff_next = fixed["timestamp"] - next_behind["timestamp"]
            if diff_next >= 0:
                behind = next_behind
                continue

            best = behind if diff_behind <= abs(diff_next) else next_behind
            car_match, infra_match = (fixed, best) if fixed_is_car else (best, fixed)
            pairs.append((_project_sample(car_match), _project_sample(infra_match)))

            new_fixed = self._next_lidar_sample(fixed, by_token)
            if new_fixed is None:
                break

            if new_fixed["timestamp"] < next_behind["timestamp"]:
                fixed, behind, fixed_is_car = next_behind, new_fixed, not fixed_is_car
            else:
                fixed, behind = new_fixed, next_behind

        return pairs

    @staticmethod
    def _next_lidar_sample(sample: dict, by_token: dict) -> Optional[dict]:
        """Follow the ``next`` token to the following LiDAR sample, if any."""
        next_token = sample["next"]
        if not next_token:
            return None

        return by_token[next_token]
