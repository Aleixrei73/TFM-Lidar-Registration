"""Identity point cloud registration method."""

from __future__ import annotations

import logging

import numpy as np

from .base import PointCloud, RegistrationMethod, Transform

logger = logging.getLogger(__name__)


class IdentityMethod(RegistrationMethod):
    """Point cloud registration that returns the given initial transform as-is.

    Useful as a no-op baseline: it performs no alignment of its own and
    simply passes through the `transform_imu` guess it is given.
    """

    def fit(self, source: PointCloud, target: PointCloud, **kwargs) -> Transform:
        """Return `transform_imu` from `kwargs` unchanged.

        Parameters
        ----------
        source : PointCloud
            Point cloud to be aligned onto `target`, shape (N, 3).
        target : PointCloud
            Reference point cloud, shape (M, 3).
        **kwargs
            Must contain `transform_imu`, the 4x4 transform to return.

        Returns
        -------
        np.ndarray
            The `transform_imu` transform passed in `kwargs`, unchanged.
        """
        return np.asarray(kwargs["transform_imu"])
