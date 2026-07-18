"""
V2X-Real Dataset for Point Cloud Registration
==============================================
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.optimize import brentq
from scipy.special import ive
from torch.utils.data import Dataset
import logging

from utils.helpers import x1_to_x2

# A pair entry: ((src_agent_dir, src_timestamp), (tgt_agent_dir, tgt_timestamp))
PairEntry = Tuple[Tuple[Path, str], Tuple[Path, str]]

logger = logging.getLogger(__name__)

def read_point_cloud(path: Path) -> np.ndarray:
    """
    Read a V2X-Real ``.bin`` point cloud file.

    Parameters
    ----------
    path : Path
        Path to a ``.bin`` file storing a flat float32 array with 4
        columns (x, y, z, intensity).

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) and dtype float32 with the point XYZ
        coordinates. The intensity column is discarded.
    """
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def kappa_from_circular_std(sigma_deg: float) -> float:
    """
    Find the von Mises concentration matching a target circular std.

    Parameters
    ----------
    sigma_deg : float
        Desired circular standard deviation, in degrees. Must be > 0.

    Returns
    -------
    float
        The concentration ``kappa`` such that a ``von Mises(0, kappa)``
        sample has circular standard deviation ``sigma_deg``.
    """
    sigma_rad = np.radians(sigma_deg)
    target_r = np.exp(-(sigma_rad**2) / 2)

    def mean_resultant_length(kappa: float) -> float:
        # ive is the exponentially-scaled Bessel function (ive(v, z) =
        # iv(v, z) * exp(-z)); the scaling cancels in the ratio, avoiding
        # the inf/inf overflow plain iv hits for large kappa.
        return ive(1, kappa) / ive(0, kappa) - target_r

    # Bracket scaled off the small-noise approximation (kappa ~ 1/sigma^2),
    # with margin since the true kappa can run higher for large sigma_deg.
    upper = max(10.0, 10.0 / sigma_rad**2)

    return brentq(mean_resultant_length, 0.0, upper)


def add_pose_noise(
    pose: List[float], position_noise: float, kappa: Optional[float]
) -> List[float]:
    """
    Perturb a pose with Gaussian position noise and von Mises angular noise.

    Parameters
    ----------
    pose : list of float
        ``[x, y, z, roll, yaw, pitch]``, positions in meters and angles in
        degrees.
    position_noise : float
        Standard deviation, in meters, of the zero-mean Gaussian noise
        added to x, y and z.
    kappa : float or None
        Precomputed von Mises concentration parameter (see
        ``kappa_from_circular_std``) shared by roll, yaw and pitch noise.
        ``None`` disables angular noise.

    Returns
    -------
    list of float
        The noisy pose, in the same ``[x, y, z, roll, yaw, pitch]`` layout.
    """
    x, y, z, roll, yaw, pitch = pose

    x += np.random.normal(0.0, position_noise)
    y += np.random.normal(0.0, position_noise)
    z += np.random.normal(0.0, position_noise)

    if kappa is not None:
        roll += np.degrees(np.random.vonmises(0.0, kappa))
        yaw += np.degrees(np.random.vonmises(0.0, kappa))
        pitch += np.degrees(np.random.vonmises(0.0, kappa))

    return [x, y, z, roll, yaw, pitch]


def build_pairs(root_dir: str) -> List[PairEntry]:
    """
    Build all infrastructure-vehicle agent pairs sharing a timestamp.

    Parameters
    ----------
    root_dir : str
        Path to a dataset split folder (e.g. ``V2X-Real-Lidar-64/train``),
        containing one subfolder per scenario, each holding one subfolder
        per agent named after the agent id (negative for infrastructure,
        non-negative for vehicles).

    Returns
    -------
    list of ((Path, str), (Path, str))
        One entry per (negative agent folder, positive agent folder,
        shared timestamp) triple. Each entry is
        ``((src_agent_dir, timestamp), (tgt_agent_dir, timestamp))``,
        where ``src_agent_dir`` is an infrastructure agent folder and
        ``tgt_agent_dir`` is a vehicle agent folder. Pairs are ordered
        scenario by scenario, then by negative folder, then by positive
        folder, then by ascending timestamp.
    """
    pairs: List[PairEntry] = []

    for scenario_dir in sorted(Path(root_dir).iterdir()):
        if not scenario_dir.is_dir():
            continue

        neg_dirs, pos_dirs = [], []
        for agent_dir in sorted(scenario_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            try:
                agent_id = int(agent_dir.name)
            except ValueError:
                continue
            (neg_dirs if agent_id < 0 else pos_dirs).append(agent_dir)

        for neg_dir in neg_dirs:
            neg_ts = {p.stem for p in neg_dir.glob("*.yaml")}
            for pos_dir in pos_dirs:
                pos_ts = {p.stem for p in pos_dir.glob("*.yaml")}
                for ts in sorted(neg_ts & pos_ts):
                    pairs.append(((neg_dir, ts), (pos_dir, ts)))

    return pairs


class V2XReal(Dataset):
    """
    PyTorch Dataset for point cloud registration on V2X-Real.

    Parameters
    ----------
    root_dir : str
        Path to a dataset split folder (e.g. ``v2xreal/train``).
    """

    def __init__(self, root_dir: str, position_noise : float = 0, angular_noise : float = 0) -> None:
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        if not root_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {root_dir}")
    
        self.root_dir = root_dir
        self.position_noise = position_noise
        self.angular_noise = angular_noise
        # Precomputed once since it only depends on angular_noise, which is
        # fixed for the dataset's lifetime.
        self.kappa = (
            kappa_from_circular_std(angular_noise) if angular_noise > 0 else None
        )
        self.pairs = build_pairs(self.root_dir)

        logger.info(f"Using positional noise : {position_noise} and angular noise : {angular_noise}")

    def __getitem__(self, idx: int) -> Dict:
        """
        Load a registration sample.

        Parameters
        ----------
        idx : int
            Index into ``self.pairs``.

        Returns
        -------
        dict
            ``infra_pts`` / ``vehicle_pts`` : (N, 3) float32 tensors with
            raw point clouds. ``T_vehicle_to_infra`` / ``T_infra_to_vehicle``
            : (4, 4) float32 tensors with the rigid transforms between the
            two LiDAR frames. ``meta`` : dict with scenario, timestamps and
            agent ids.
        """
        (infra_dir, infra_ts), (vehicle_dir, vehicle_ts) = self.pairs[idx]

        infra_pts = read_point_cloud(infra_dir / f"{infra_ts}.bin")
        vehicle_pts = read_point_cloud(vehicle_dir / f"{vehicle_ts}.bin")

        infra_imu_pose = load_yaml(infra_dir / f"{infra_ts}.yaml")["lidar_pose"]
        vehicle_imu_pose = load_yaml(vehicle_dir / f"{vehicle_ts}.yaml")["lidar_pose"]
        vehicle_imu_pose = add_pose_noise(
            vehicle_imu_pose, self.position_noise, self.kappa
        )

        infra_true_pose = load_yaml(infra_dir / f"{infra_ts}.yaml")["true_ego_pose"]
        vehicle_true_pose = load_yaml(vehicle_dir / f"{vehicle_ts}.yaml")[
            "true_ego_pose"
        ]

        # x1_to_x2 maps points from x1's local frame into x2's local frame
        T_vehicle_to_infra_imu = x1_to_x2(vehicle_imu_pose, infra_imu_pose)
        T_infra_to_vehicle_imu = x1_to_x2(infra_imu_pose, vehicle_imu_pose)
        T_vehicle_to_infra = x1_to_x2(vehicle_true_pose, infra_true_pose)
        T_infra_to_vehicle = x1_to_x2(infra_true_pose, vehicle_true_pose)

        return {
            "infra_pts": torch.from_numpy(infra_pts),
            "vehicle_pts": torch.from_numpy(vehicle_pts),
            "T_vehicle_to_infra": torch.from_numpy(T_vehicle_to_infra).float(),
            "T_infra_to_vehicle": torch.from_numpy(T_infra_to_vehicle).float(),
            "T_vehicle_to_infra_imu": torch.from_numpy(T_vehicle_to_infra_imu).float(),
            "T_infra_to_vehicle_imu": torch.from_numpy(T_infra_to_vehicle_imu).float(),
            "meta": {
                "scenario": infra_dir.parent.name,
                "infra_timestamp": infra_ts,
                "vehicle_timestamp": vehicle_ts,
                "infra_agent": infra_dir.name,
                "vehicle_agent": vehicle_dir.name,
            },
        }

    def __len__(self):
        return len(self.pairs)
