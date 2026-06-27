"""Voxelized GICP (VGICP) based point cloud registration method."""

from __future__ import annotations

import logging

import numpy as np
import small_gicp

from .base import PointCloud, RegistrationMethod, Transform

logger = logging.getLogger(__name__)


class VGICP(RegistrationMethod):
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

    def fit(self, source: PointCloud, target: PointCloud) -> Transform:
        """Estimate the transform that aligns `source` onto `target` via VGICP.

        Parameters
        ----------
        source : PointCloud
            Point cloud to be aligned onto `target`, shape (N, 3).
        target : PointCloud
            Reference point cloud, shape (M, 3).

        Returns
        -------
        np.ndarray
            The estimated 4x4 homogeneous transform mapping `source` onto
            `target`.
        """
        # preprocess_points attaches per-point covariances, required by the
        # GICP-family cost; the returned KdTree is unused since VGICP looks
        # up correspondences in the target voxel map instead.
        result = small_gicp.align(
            target,  # raw Nx3 numpy array
            source,  # raw Mx3 numpy array
            registration_type="VGICP",
            downsampling_resolution=self.voxel_size,
            num_threads=self.num_threads,
        )

        if not result.converged:
            logger.warning("VGICP did not converge; returning best-effort transform.")

        return np.asarray(result.T_target_source)
