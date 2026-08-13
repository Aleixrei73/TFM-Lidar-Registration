"""Voxelized GICP (VGICP) based point cloud registration method."""

from __future__ import annotations

import logging

import numpy as np
import small_gicp

from .base import PointCloud, RegistrationMethod, Transform

logger = logging.getLogger(__name__)


class VGICPPlaneInit(RegistrationMethod):
    """Point cloud registration using Voxelized GICP.

    VGICP replaces the per-point covariance lookups of plain GICP with a
    Gaussian voxel map built from the target cloud, trading some accuracy
    for substantially faster nearest-neighbor lookups on large clouds.
    """

    def __init__(
        self,
        voxel_size: float,
        num_threads: int = 4,
        voxel_resolution: float = 4.0,
        use_initialization: bool = False,
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
        self.voxel_resolution = voxel_size * voxel_resolution
        self.use_initialization = use_initialization

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

        source_plane = source.copy()
        target_plane = target.copy()

        source_plane[:, 2] = 0.0
        target_plane[:, 2] = 0.0

        result = small_gicp.align(
            target_plane,  # raw Nx3 numpy array
            source_plane,  # raw Mx3 numpy array
            init_T_target_source=transform_imu,
            registration_type="VGICP",
            downsampling_resolution=self.voxel_size,
            voxel_resolution = self.voxel_resolution,
            num_threads=self.num_threads,
        )

        result = small_gicp.align(
            target,  # raw Nx3 numpy array
            source,  # raw Mx3 numpy array
            init_T_target_source=result.T_target_source,
            registration_type="VGICP",
            downsampling_resolution=self.voxel_size,
            voxel_resolution = self.voxel_resolution,
            num_threads=self.num_threads,
        )

        if not result.converged:
            logger.warning("VGICP did not converge; returning best-effort transform.")

        return np.asarray(result.T_target_source)
