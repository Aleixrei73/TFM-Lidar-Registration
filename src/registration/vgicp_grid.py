"""Voxelized GICP (VGICP) based point cloud registration method."""

from __future__ import annotations

import logging

import numpy as np
import small_gicp
from scipy.spatial import KDTree
import open3d as o3d
import time

from .base import PointCloud, RegistrationMethod, Transform

logger = logging.getLogger(__name__)

def make_transform(gps_pose, dx, dy, dtheta_deg):
    c, s = np.cos(np.radians(dtheta_deg)), np.sin(np.radians(dtheta_deg))
    delta = np.array([
        [c, -s, 0, dx],
        [s,  c, 0, dy],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    return gps_pose @ delta

def voxel_downsample_o3d(points, voxel_size):
    pc = o3d.t.geometry.PointCloud(o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32))
    pc_ds = pc.voxel_down_sample(voxel_size)
    return pc_ds.point.positions.numpy()

def build_transforms(gps_pose, t_range, t_step, yaw_range, yaw_step):
    dxs = np.arange(-t_range, t_range + 1e-9, t_step)
    dys = np.arange(-t_range, t_range + 1e-9, t_step)
    dths = np.arange(-yaw_range, yaw_range + 1e-9, yaw_step)
    return [make_transform(gps_pose, dx, dy, dth)
            for dx in dxs for dy in dys for dth in dths]

def grid_search_batched(source_ds, tree, transforms, max_corr_dist):
    n_candidates = len(transforms)
    n_points = len(source_ds)

    all_transformed = np.empty((n_candidates, n_points, 3), dtype=np.float32)
    for i, T in enumerate(transforms):
        R, t = T[:3, :3], T[:3, 3]
        all_transformed[i] = source_ds @ R.T + t

    flat = all_transformed.reshape(-1, 3)
    dists, _ = tree.query(flat, distance_upper_bound=max_corr_dist, workers=-1)
    dists = dists.reshape(n_candidates, n_points)
    valid = np.isfinite(dists)

    counts = valid.sum(axis=1)
    fitness = counts / n_points
    sq = np.where(valid, dists ** 2, 0.0)
    inlier_rmse = np.sqrt(np.divide(sq.sum(axis=1), counts,
                                     out=np.zeros(n_candidates), where=counts > 0))

    scores = list(zip(fitness, -inlier_rmse))
    best_idx = max(range(n_candidates), key=lambda i: scores[i])
    return transforms[best_idx]

class VGICPGrid(RegistrationMethod):
    """Point cloud registration using Voxelized GICP.

    VGICP replaces the per-point covariance lookups of plain GICP with a
    Gaussian voxel map built from the target cloud, trading some accuracy
    for substantially faster nearest-neighbor lookups on large clouds.
    """

    def __init__(
        self,
        voxel_size: float,
        num_threads: int = 4,
        voxel_resolution: float = 1.0,
        correspondance_distance: float = 5,
        use_initialization: bool = False,
        coarse_range : float = 5.0,
        coarse_step_size : float = 1.0,
        coarse_angle_range : float = 20.0,
        coarse_angle_step_size : float = 5.0,
        fine_range : float = 1.0,
        fine_step_size : float = 0.2,
        fine_angle_range : float = 5.0,
        fine_angle_step_size : float = 1.0
    ) -> None:
        """Store the voxel resolution used for downsampling and registration.

        Parameters
        ----------
        voxel_size : float
            Voxel edge length, in the dataset's length units, used both to
            downsample the input clouds in `preprocess` and to build the
            target Gaussian voxel map in `fit`.
        num_threads : int
            Number of threads used by `small_gicp` for covariance
            estimation and alignment.
        """
        self.voxel_size = voxel_size
        self.num_threads = num_threads
        self.voxel_resolution = voxel_resolution
        self.correspondance_distance = correspondance_distance
        self.use_initialization = use_initialization

        self.coarse_range = coarse_range
        self.coarse_step_size = coarse_step_size
        self.fine_range = fine_range
        self.fine_step_size = fine_step_size

        self.coarse_angle_range = coarse_angle_range
        self.coarse_angle_step_size = coarse_angle_step_size
        self.fine_angle_range = fine_angle_range
        self.fine_angle_step_size = fine_angle_step_size

    def fit(self, source: PointCloud, target: PointCloud, **kwargs) -> Transform:
        """Estimate the transform that aligns `source` onto `target` via VGICP.

        Parameters
        ----------
        source : PointCloud
            Point cloud to be aligned onto `target`, shape (N, 3).
        target : PointCloud
            Reference point cloud, shape (M, 3).
        **kwargs
            May contain `transform_imu`, a 4x4 initial transform guess used
            to seed the alignment. Falls back to the identity transform
            when not provided.

        Returns
        -------
        np.ndarray
            The estimated 4x4 homogeneous transform mapping `source` onto
            `target`.
        """
        transform_imu = (
            kwargs["transform_imu"] if self.use_initialization else np.eye(4)
        )

        coarse_transforms = build_transforms(transform_imu, self.coarse_range, self.coarse_step_size, self.coarse_angle_range, self.coarse_angle_step_size)   # 1,089
        fine_transforms    = None  # built after coarse result, centered on it

        coarse_voxel = 1 * 2.0   # coarse just needs rough ranking → aggressive downsample
        fine_voxel   = 1


        target_donwsample = voxel_downsample_o3d(target, coarse_voxel)
        source_donwsample = voxel_downsample_o3d(source, coarse_voxel)

        tree = KDTree(target_donwsample)

        coarse_T = grid_search_batched(source_donwsample, tree, coarse_transforms, 1)

        fine_transforms = build_transforms(coarse_T, self.fine_range, self.fine_step_size, self.fine_angle_range, self.fine_angle_step_size)


        target_donwsample = voxel_downsample_o3d(target, fine_voxel)
        source_donwsample = voxel_downsample_o3d(source, fine_voxel)

        tree = KDTree(target_donwsample)

        fine_T = grid_search_batched(source_donwsample, tree, fine_transforms, 0.5)

        result = small_gicp.align(
            target,  # raw Nx3 numpy array
            source,  # raw Mx3 numpy array
            init_T_target_source=fine_T,
            registration_type="VGICP",
            downsampling_resolution=self.voxel_size,
            num_threads=self.num_threads,
        )


        if not result.converged:
            logger.warning("VGICP did not converge; returning best-effort transform.")

        return np.asarray(result.T_target_source)
