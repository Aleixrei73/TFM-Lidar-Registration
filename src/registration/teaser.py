"""TEASER++ based point cloud registration method."""

from __future__ import annotations

import numpy as np
import teaserpp_python

from .base import PointCloud, RegistrationMethod


class TeaserRegistration(RegistrationMethod):
    """Point cloud registration using the TEASER++ robust solver.

    TEASER++ recovers a (possibly scaled) rigid transform between two point
    clouds via a truncated least-squares formulation solved with Graduated
    Non-Convexity (GNC), which makes it robust to a high fraction of outlier
    correspondences.
    """

    def __init__(
        self,
        cbar2: float = 1,
        noise_bound: float = 0.01,
        estimate_scaling: bool = True,
        rotation_estimation_algorithm: teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        ),
        rotation_gnc_factor: float = 1.4,
        rotation_max_iterations: int = 100,
        rotation_cost_threshold: float = 1e-12,
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
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()

        solver_params.cbar2 = cbar2
        solver_params.noise_bound = noise_bound
        solver_params.estimate_scaling = estimate_scaling
        solver_params.rotation_estimation_algorithm = rotation_estimation_algorithm
        solver_params.rotation_gnc_factor = rotation_gnc_factor
        solver_params.rotation_max_iterations = rotation_max_iterations
        solver_params.rotation_cost_threshold = rotation_cost_threshold

        self.solver_params = solver_params
        self.solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    def fit(self, source: PointCloud, target: PointCloud) -> np.ndarray:
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
        self.solver.solve(source.T, target.T)
        solution = self.solver.getSolution()

        transform = np.eye(4)
        transform[:3, :3] = solution.scale * solution.rotation
        transform[:3, 3] = solution.translation.flatten()

        return transform
