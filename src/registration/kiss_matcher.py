"""TEASER++ based point cloud registration method."""

from __future__ import annotations

import numpy as np
import kiss_matcher

from .base import PointCloud, RegistrationMethod, Transform

class KISSMatcher(RegistrationMethod):
    """Point cloud registration using the TEASER++ robust solver.

    TEASER++ recovers a (possibly scaled) rigid transform between two point
    clouds via a truncated least-squares formulation solved with Graduated
    Non-Convexity (GNC), which makes it robust to a high fraction of outlier
    correspondences.
    """

    def __init__(
        self,
        voxel_size: float = 0.5
    ) -> None:
        """Build the TEASER++ solver parameters.

        Parameters
        ----------
        cbar2 : float
            Upper bound on the normalized residual (in noise-bound units)
            used by the truncated least-squares cost; correspondences with a
            larger residual are treated as outliers.
        noise_bound : float
            Expected noise level of the input points, in the dataset's
            length units. Scales both the truncated least-squares cost and
            the outlier rejection threshold.
        estimate_scaling : bool
            Whether to additionally estimate a similarity scale factor
            between `source` and `target`, instead of assuming a rigid
            (scale=1) transform.
        rotation_estimation_algorithm : teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM
            Algorithm used to solve the rotation subproblem. Defaults to
            Graduated Non-Convexity with Truncated Least Squares (GNC_TLS)
            for robustness to outlier correspondences.
        rotation_gnc_factor : float
            Multiplicative factor that grows the GNC control parameter at
            each iteration, controlling how fast the solver anneals from a
            convex relaxation to the non-convex truncated cost.
        rotation_max_iterations : int
            Hard cap on the number of GNC iterations for the rotation
            estimation step, bounding worst-case runtime.
        rotation_cost_threshold : float
            Convergence tolerance on the rotation cost; iteration stops
            early once the change in cost between iterations falls below
            this value.
        """
        config = kiss_matcher.KISSMatcherConfig(voxel_size)
        self.matcher = kiss_matcher.KISSMatcher(config)


    def fit(self, source: PointCloud, target: PointCloud) -> Transform:
        """Estimate the transform that aligns `source` onto `target`.

        Parameters
        ----------
        source : PointCloud
            Point cloud to be aligned onto `target`, shape (N, 3).
        target : PointCloud
            Reference point cloud, shape (N, 3).

        Returns
        -------
        np.ndarray
            The estimated 4x4 homogeneous transform mapping `source` onto
            `target`.
        """

        # TEASER++ expects points as 3xN arrays, columns being individual points
        self.matcher.reset()
        result = self.matcher.estimate(source, target)
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(result.rotation)
        transform[:3, 3] = np.asarray(result.translation).flatten()

        return transform
