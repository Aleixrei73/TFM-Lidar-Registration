"""
V2X-Real Dataset for Point Cloud Registration
==============================================

Dataset structure (OPV2V format):

    v2xreal/
    ├── train/
    │   └── 2023-03-17-15-53-02_1_0/       # scenario folder
    │       ├── 0/                           # CAV agent  (positive id)
    │       │   ├── 00000.pcd
    │       │   ├── 00000.yaml
    │       │   └── ...
    │       └── -1/                          # infrastructure (negative id)
    │           ├── 00000.pcd
    │           ├── 00000.yaml
    │           └── ...
    ├── validate/
    └── test/

Each per-timestamp YAML contains:
    lidar_pose: [x, y, z, roll, yaw, pitch]  # LiDAR pose in world frame (degrees)

Each sample returns:
    src_pts : (N, 3) float32  – raw source point cloud
    tgt_pts : (M, 3) float32  – raw target point cloud
    T_gt    : (4, 4) float32  – rigid transform mapping source → target frame
    meta    : dict            – scenario, timestamp, agent IDs
"""

import os
import glob
import math
from pathlib import Path
from typing import List, Dict

import numpy as np
import yaml
import open3d as o3d
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_point_cloud(path: str) -> np.ndarray:
    """
    Read a point cloud file and return its XYZ coordinates.

    Parameters
    ----------
    path : str
        Path to a ``.pcd`` file.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) and dtype float32 with the point XYZ
        coordinates. Any additional fields (e.g. intensity) are discarded.
    """
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float32)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def pose_to_transform(pose: List[float]) -> np.ndarray:
    """
    Convert an OPV2V/CARLA pose to a 4x4 homogeneous transform.

    Parameters
    ----------
    pose : list of float
        ``[x, y, z, roll, yaw, pitch]`` in metres and degrees, as stored
        in the per-timestamp YAML files (note: yaw precedes pitch, per
        the CARLA/OpenCOOD convention).

    Returns
    -------
    np.ndarray
        4x4 float32 transform T_world_agent, mapping points from the
        agent's local LiDAR frame into the CARLA world frame.

    Notes
    -----
    CARLA's world frame is left-handed, so the rotation block below is
    not the standard right-handed ZYX Euler product; it follows the
    formula used by OpenCOOD's ``x_to_world``.
    """
    x, y, z, roll, yaw, pitch = pose[:6]
    c_y, s_y = math.cos(math.radians(yaw)),   math.sin(math.radians(yaw))
    c_r, s_r = math.cos(math.radians(roll)),  math.sin(math.radians(roll))
    c_p, s_p = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))

    R = np.array([
        [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r],
        [s_p,       -c_p * s_r,                   c_p * c_r                 ],
    ])

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3,  3] = [x, y, z]
    return T


def relative_transform(T_src_world: np.ndarray,
                        T_tgt_world: np.ndarray) -> np.ndarray:
    """
    Compose two world-frame transforms into a source-to-target transform.

    Parameters
    ----------
    T_src_world : np.ndarray
        4x4 transform mapping the source agent's local frame to the
        world frame.
    T_tgt_world : np.ndarray
        4x4 transform mapping the target agent's local frame to the
        world frame.

    Returns
    -------
    np.ndarray
        4x4 float32 transform ``T_gt = T_tgt_world^{-1} @ T_src_world``
        that maps points from the source frame into the target frame.
    """
    T_tgt_inv = np.linalg.inv(T_tgt_world.astype(np.float64))
    return (T_tgt_inv @ T_src_world.astype(np.float64)).astype(np.float32)


# ---------------------------------------------------------------------------
# Scenario database
# ---------------------------------------------------------------------------

def build_scenario_db(root_dir: str) -> List[Dict]:
    """
    Index a dataset split into per-pair, per-timestamp sample descriptors.

    Infrastructure agents have negative integer IDs (e.g. '-1').
    CAV agents have non-negative integer IDs (e.g. '0', '1').

    Parameters
    ----------
    root_dir : str
        Path to a split folder (e.g. ``v2xreal/train``) containing one
        subfolder per scenario, each holding one subfolder per agent.

    Returns
    -------
    list of dict
        One descriptor per (agent_A, agent_B, timestamp) triple that
        exists for both agents, with keys ``scenario``, ``timestamp``,
        ``src_agent``, ``tgt_agent``, ``src_pcd``, ``tgt_pcd``,
        ``src_yaml``, ``tgt_yaml``.
    """
    samples: List[Dict] = []

    for scenario_dir in sorted(Path(root_dir).iterdir()):
        if not scenario_dir.is_dir():
            continue

        # Collect timestamps available per agent
        agent_ts: Dict[str, Dict[str, Dict]] = {}
        for agent_dir in sorted(scenario_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            ts_map = {}
            for yaml_path in sorted(glob.glob(str(agent_dir / "*.yaml"))):
                ts = Path(yaml_path).stem          # e.g. "00000"
                pcd_path = str(agent_dir / f"{ts}.pcd")
                if os.path.exists(pcd_path):
                    ts_map[ts] = {"pcd": pcd_path, "yaml": yaml_path}
            if ts_map:
                agent_ts[agent_dir.name] = ts_map

        # Create one sample per (agent pair, shared timestamp)
        agent_ids = list(agent_ts.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                src_id, tgt_id = agent_ids[i], agent_ids[j]
                for ts in sorted(set(agent_ts[src_id]) & set(agent_ts[tgt_id])):
                    samples.append({
                        "scenario":  scenario_dir.name,
                        "timestamp": ts,
                        "src_agent": src_id,
                        "tgt_agent": tgt_id,
                        "src_pcd":   agent_ts[src_id][ts]["pcd"],
                        "tgt_pcd":   agent_ts[tgt_id][ts]["pcd"],
                        "src_yaml":  agent_ts[src_id][ts]["yaml"],
                        "tgt_yaml":  agent_ts[tgt_id][ts]["yaml"],
                    })
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class V2XRealRegistrationDataset(Dataset):
    """
    PyTorch Dataset for point cloud registration on V2X-Real.

    Parameters
    ----------
    root_dir : str
        Path to a split folder, e.g. ``v2xreal/train``.
    source_type : str
        Agent role for the source cloud: ``"vehicle"``, ``"infrastructure"``,
        or ``"any"``.
    target_type : str
        Agent role for the target cloud: same options.
    """

    def __init__(
        self,
        root_dir: str,
        source_type: str = "infrastructure",
        target_type: str = "vehicle",
    ):
        assert os.path.isdir(root_dir), f"Directory not found: {root_dir}"
        assert source_type in ("vehicle", "infrastructure", "any")
        assert target_type in ("vehicle", "infrastructure", "any")

        self.root_dir    = root_dir
        self.source_type = source_type
        self.target_type = target_type
        self.samples     = self._filter(build_scenario_db(root_dir))

        if not self.samples:
            raise RuntimeError(
                f"No samples found in '{root_dir}' for "
                f"source='{source_type}', target='{target_type}'."
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _is_vehicle(agent_id: str) -> bool:
        try:
            return int(agent_id) >= 0
        except ValueError:
            return False

    def _matches(self, agent_id: str, role: str) -> bool:
        if role == "any":
            return True
        return (role == "vehicle") == self._is_vehicle(agent_id)

    def _filter(self, samples: List[Dict]) -> List[Dict]:
        out = []
        for s in samples:
            if self._matches(s["src_agent"], self.source_type) and \
               self._matches(s["tgt_agent"], self.target_type):
                out.append(s)
            # Also accept the swapped direction
            elif self.source_type != self.target_type and \
                 self._matches(s["tgt_agent"], self.source_type) and \
                 self._matches(s["src_agent"], self.target_type):
                out.append({**s,
                    "src_agent": s["tgt_agent"], "tgt_agent": s["src_agent"],
                    "src_pcd":   s["tgt_pcd"],   "tgt_pcd":   s["src_pcd"],
                    "src_yaml":  s["tgt_yaml"],  "tgt_yaml":  s["src_yaml"],
                })
        return out

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (f"V2XRealRegistrationDataset(root='{self.root_dir}', "
                f"src='{self.source_type}', tgt='{self.target_type}', "
                f"n={len(self)})")

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        src_pts = read_point_cloud(s["src_pcd"])   # (N, 3)
        tgt_pts = read_point_cloud(s["tgt_pcd"])   # (M, 3)

        src_meta = load_yaml(s["src_yaml"])
        tgt_meta = load_yaml(s["tgt_yaml"])

        T_src = pose_to_transform(src_meta["lidar_pose"])
        T_tgt = pose_to_transform(tgt_meta["lidar_pose"])
        T_gt  = relative_transform(T_src, T_tgt)   # (4, 4)

        return {
            "src_pts": torch.from_numpy(src_pts),
            "tgt_pts": torch.from_numpy(tgt_pts),
            "T_gt":    torch.from_numpy(T_gt),
            "meta": {
                "scenario":  s["scenario"],
                "timestamp": s["timestamp"],
                "src_agent": s["src_agent"],
                "tgt_agent": s["tgt_agent"],
            },
        }