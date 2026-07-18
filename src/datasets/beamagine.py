"""
Beamagine Dataset for Point Cloud Registration
===============================================
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.helpers import gps_to_enu, heading_to_yaw, pose_to_matrix

MIN_SCENE_NUMBER = 1
MAX_SCENE_NUMBER = 7

# Fields kept from a sample_data entry once it is matched into a pair.
SAMPLE_FIELDS = ("filename", "calibrated_sensor_token", "timestamp", "sample_token")

GNSS_SUBDIR = Path("GNSS_car") / "INS"
LIDAR_SUBDIR = Path("Samples") / "LIDAR"

# Raw point fields are int32 [x_mm, y_mm, z_mm, intensity, <unused>].
POINT_FIELD_COUNT = 5
POINT_XYZ_SCALE_M = 1000.0


def load_json(path: Path) -> list:
    """Load a JSON file into a list of dicts."""
    with open(path, "r") as f:
        return json.load(f)


def read_point_cloud(path: Path) -> np.ndarray:
    """
    Read a Beamagine ``.bin`` point cloud file.

    Parameters
    ----------
    path : Path
        Path to a ``.bin`` file: a ``uint32`` point-count header followed
        by that many ``int32`` quintuples ``(x_mm, y_mm, z_mm, intensity,
        <unused>)``.

    Returns
    -------
    np.ndarray
        Array of shape (N, 4) and dtype float64 with columns
        ``(x, y, z, intensity)``. ``x``, ``y``, ``z`` are converted from
        millimeters to meters; intensity is left unscaled. Points with a
        NaN in ``x``, ``y``, or ``z`` are dropped.
    """
    with open(path, "rb") as f:
        point_count = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        raw = np.fromfile(f, dtype=np.int32).reshape(point_count, POINT_FIELD_COUNT)

    points = raw[:, :4].astype(np.float64)
    points[:, :3] /= POINT_XYZ_SCALE_M

    valid = ~np.isnan(points[:, :3]).any(axis=1)

    return points[valid]


def _project_sample(sample: dict) -> dict:
    """Reduce a sample_data entry to the fields needed downstream."""
    return {field: sample[field] for field in SAMPLE_FIELDS}


def _dms_to_decimal(degrees: float, minutes: float, seconds: float) -> float:
    """Convert a degrees/minutes/seconds angle to decimal degrees."""
    return degrees + minutes / 60 + seconds / 3600


# Fixed infrastructure LiDAR pose (surveyed, constant across all scenes).
INFRA_LAT_DEG = _dms_to_decimal(41, 33, 48)
INFRA_LON_DEG = _dms_to_decimal(2, 1, 15)
INFRA_ALT_M = 339
INFRA_YAW_DEG = 39
INFRA_PITCH_DEG = -9
INFRA_ROLL_DEG = 0.0

# Raw Beamagine point clouds sit in the sensor's own optical-style frame
# (X=right, Y=down, Z=forward/depth), inferred from point-spread statistics
# (Z is always positive with a ~140 m range on both car and infra clouds --
# not a plausible vertical extent). This rotates raw sensor points into the
# FLU/ENU body frame (X=forward, Y=left, Z=up) that pose_to_matrix assumes.
# Not from a confirmed spec/calibration file; revisit if one becomes available.
SENSOR_TO_BODY_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


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
        self.gnss_records = self._load_gnss_records()
        self._attach_gps()
        self._attach_transforms_test()

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load a registration sample.

        Parameters
        ----------
        idx : int
            Index into ``self.sample_pairs``.

        Returns
        -------
        dict
            ``vehicle_pts`` / ``infra_pts`` : (N, 3) float64 tensors with
            raw point cloud coordinates (intensity discarded).
            ``T_vehicle_to_infra`` / ``T_infra_to_vehicle`` : (4, 4)
            float32 tensors with the rigid transform between the two
            LiDAR frames. ``T_vehicle_to_infra_imu`` /
            ``T_infra_to_vehicle_imu`` mirror the same transform (Beamagine
            has no separate IMU-based pose).
        """
        car_sample, infra_sample = self.sample_pairs[idx]

        car_pts = read_point_cloud(
            self.root_dir / LIDAR_SUBDIR / car_sample["filename"]
        )
        infra_pts = read_point_cloud(
            self.root_dir / LIDAR_SUBDIR / infra_sample["filename"]
        )

        return {
            "vehicle_pts": torch.from_numpy(car_pts[:, :3]),
            "infra_pts": torch.from_numpy(infra_pts[:, :3]),
            "T_vehicle_to_infra": torch.from_numpy(
                car_sample["T_car_to_fixed"]
            ).float(),
            "T_infra_to_vehicle": torch.from_numpy(
                infra_sample["T_fixed_to_car"]
            ).float(),
            "T_vehicle_to_infra_imu": torch.from_numpy(
                car_sample["T_car_to_fixed"]
            ).float(),
            "T_infra_to_vehicle_imu": torch.from_numpy(
                infra_sample["T_fixed_to_car"]
            ).float(),
        }

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
            d["token"]: d for d in load_json(self.json_dir / "sample_data.json")
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

    def _load_gnss_records(self) -> List[dict]:
        """
        Load the car's INS log for this scene.

        Returns
        -------
        list of dict
            Rows of the scene's ``GNSS_car/INS`` csv file, ordered by
            ascending ``unix_time_ns``, with the relevant fields cast to
            ``float``/``int``.
        """
        gnss_files = sorted((self.root_dir / GNSS_SUBDIR).glob("*.csv"))
        gnss_path = gnss_files[self.scene_number]

        with open(gnss_path, "r") as f:
            reader = csv.DictReader(f)
            records = [
                {
                    "lat_deg": float(row["lat_deg"]),
                    "lon_deg": float(row["lon_deg"]),
                    "alt_m": float(row["alt_m"]),
                    "roll_deg": float(row["roll_deg"]),
                    "pitch_deg": float(row["pitch_deg"]),
                    "heading_deg": float(row["heading_deg"]),
                    "unix_time_ns": int(row["unix_time_ns"]),
                }
                for row in reader
            ]

        return records

    def _attach_gps(self) -> None:
        """
        Attach GPS position and orientation to every pair in ``self.sample_pairs``.

        The infrastructure LiDAR uses a fixed, surveyed pose. The car's
        pose is looked up from ``self.gnss_records`` by finding, for
        each car sample's timestamp, the closest GNSS reading in time.
        Both sequences are chronologically ordered, so a single pointer
        walks ``self.gnss_records`` forward across the pairs.
        """
        gnss_idx = 0
        last_idx = len(self.gnss_records) - 1

        for car_sample, infra_sample in self.sample_pairs:
            infra_sample.update(
                lat_deg=INFRA_LAT_DEG,
                lon_deg=INFRA_LON_DEG,
                alt_m=INFRA_ALT_M,
                roll_deg=INFRA_ROLL_DEG,
                pitch_deg=INFRA_PITCH_DEG,
                yaw_deg=INFRA_YAW_DEG,
            )

            car_time_ms = car_sample["timestamp"]
            while (
                gnss_idx < last_idx
                and self.gnss_records[gnss_idx + 1]["unix_time_ns"] / 1e6 <= car_time_ms
            ):
                gnss_idx += 1

            candidates = [gnss_idx] + ([gnss_idx + 1] if gnss_idx < last_idx else [])
            best_idx = min(
                candidates,
                key=lambda i: abs(
                    self.gnss_records[i]["unix_time_ns"] / 1e6 - car_time_ms
                ),
            )
            match = self.gnss_records[best_idx]

            car_sample.update(
                lat_deg=match["lat_deg"],
                lon_deg=match["lon_deg"],
                alt_m=match["alt_m"],
                roll_deg=match["roll_deg"],
                pitch_deg=match["pitch_deg"],
                yaw_deg=match["heading_deg"],
            )

    def _attach_transforms(self) -> None:
        """
        Compute the car-to-fixed and fixed-to-car transforms for every pair.

        The fixed infra pose is the origin of a local East-North-Up frame;
        the car's GPS position is converted into that frame before both
        poses are turned into homogeneous transforms and combined.
        """
        infra_yaw = heading_to_yaw(INFRA_YAW_DEG)
        infra_to_world = pose_to_matrix(
            0.0, 0.0, 0.0, INFRA_ROLL_DEG, INFRA_PITCH_DEG, infra_yaw
        )
        world_to_infra = np.linalg.inv(infra_to_world)

        for car_sample, infra_sample in self.sample_pairs:
            car_x, car_y, car_z = gps_to_enu(
                car_sample["lat_deg"],
                car_sample["lon_deg"],
                car_sample["alt_m"],
                INFRA_LAT_DEG,
                INFRA_LON_DEG,
                INFRA_ALT_M,
            )
            car_yaw = heading_to_yaw(car_sample["yaw_deg"])
            car_to_world = pose_to_matrix(
                car_x,
                car_y,
                car_z,
                car_sample["roll_deg"],
                car_sample["pitch_deg"],
                car_yaw,
            )

            car_to_fixed = world_to_infra @ car_to_world
            fixed_to_car = np.linalg.inv(car_to_fixed)

            car_sample["T_car_to_fixed"] = car_to_fixed
            infra_sample["T_fixed_to_car"] = fixed_to_car

    def _attach_transforms_test(self) -> None:
        """
        Compute yaw-only car-to-fixed and fixed-to-car transforms.

        Same GPS-to-ENU positioning as ``_attach_transforms``, but with
        roll and pitch zeroed out for both agents, to validate positioning
        and heading independently of the roll/pitch readings. The rotation
        is built in the FLU/ENU body frame and then sandwiched between
        ``SENSOR_TO_BODY_ROTATION`` and its inverse, so the resulting
        transform maps directly between the raw sensor point clouds
        returned by ``read_point_cloud`` rather than between body frames.
        """
        infra_yaw = heading_to_yaw(INFRA_YAW_DEG)
        infra_to_world = pose_to_matrix(
            0.0, 0.0, 0.0, INFRA_ROLL_DEG, INFRA_PITCH_DEG, infra_yaw
        )
        world_to_infra = np.linalg.inv(infra_to_world)

        sensor_to_body = np.eye(4)
        sensor_to_body[:3, :3] = SENSOR_TO_BODY_ROTATION

        body_to_sensor = np.eye(4)
        body_to_sensor[:3, :3] = SENSOR_TO_BODY_ROTATION.T

        for car_sample, infra_sample in self.sample_pairs:
            car_x, car_y, car_z = gps_to_enu(
                car_sample["lat_deg"],
                car_sample["lon_deg"],
                car_sample["alt_m"],
                INFRA_LAT_DEG,
                INFRA_LON_DEG,
                INFRA_ALT_M,
            )

            car_yaw = heading_to_yaw(car_sample["yaw_deg"])
            car_to_world = pose_to_matrix(
                car_x,
                car_y,
                car_z,
                car_sample["roll_deg"],
                car_sample["pitch_deg"],
                car_yaw,
            )

            car_to_fixed = (
                body_to_sensor @ world_to_infra @ car_to_world @ sensor_to_body
            )
            fixed_to_car = np.linalg.inv(car_to_fixed)

            car_sample["T_car_to_fixed"] = car_to_fixed
            infra_sample["T_fixed_to_car"] = fixed_to_car
