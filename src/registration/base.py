"""Abstract base classes for point cloud registration methods.

This module defines the common interface that every registration method
implementation must follow, so that different algorithms can be evaluated
interchangeably.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from utils.helpers import apply_transform

logger = logging.getLogger(__name__)

PointCloud = np.ndarray
Transform = np.ndarray

class RegistrationMethod(ABC):
    """Abstract base class for point cloud registration methods.

    Subclasses must implement :meth:`fit` and may override
    :meth:`preprocess` to apply method-specific preparation (e.g.
    downsampling, denoising) to the input point clouds before fitting.
    """

    def preprocess(
        self, source: PointCloud, target: PointCloud
    ) -> tuple[PointCloud, PointCloud]:
        """Prepare the input point clouds before registration.

        Default implementation is the identity function. Subclasses may
        override this to apply method-specific preprocessing.

        Parameters
        ----------
        source : PointCloud
            Point cloud to be aligned onto `target`.
        target : PointCloud
            Reference point cloud.

        Returns
        -------
        tuple[PointCloud, PointCloud]
            The preprocessed `source` and `target` point clouds.
        """
        return source, target

    @abstractmethod
    def fit(self, source: PointCloud, target: PointCloud) -> Transform:
        """Estimate the transform that aligns `source` onto `target`.

        Parameters
        ----------
        source : PointCloud
            Point cloud to be aligned onto `target`.
        target : PointCloud
            Reference point cloud.

        Returns
        -------
        Transform
            The estimated 4x4 homogeneous transform mapping `source` onto
            `target`.
        """
        raise NotImplementedError

    def register(
        self, source: PointCloud, target: PointCloud
    ) -> tuple[PointCloud, Transform]:
        """Run the full registration pipeline: preprocess, fit, apply.

        Parameters
        ----------
        source : PointCloud
            Point cloud to be aligned onto `target`.
        target : PointCloud
            Reference point cloud.

        Returns
        -------
        tuple[PointCloud, Transform]
            The registered `source` point cloud and the 4x4 transform used
            to produce it.
        """
        logger.debug("Starting registration with %s", type(self).__name__)

        preproc_source, preproc_target = self.preprocess(source, target)
        transform = self.fit(preproc_source, preproc_target)
        registered = apply_transform(preproc_source, transform)

        logger.debug("Finished registration with %s", type(self).__name__)
        return registered, transform
