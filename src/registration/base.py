"""Abstract base classes for point cloud registration methods.

This module defines the common interface that every registration method
implementation must follow, so that different algorithms can be evaluated
interchangeably.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PointCloud = np.ndarray


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
            The estimated transform mapping `source` onto `target`.
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
            The registered `source` point cloud and the transform used to
            produce it.
        """
        logger.debug("Starting registration with %s", type(self).__name__)

        preproc_source, preproc_target = self.preprocess(source, target)
        transform = self.fit(preproc_source, preproc_target)
        registered = transform.apply_to_points(preproc_source)

        logger.debug("Finished registration with %s", type(self).__name__)
        return registered, transform


class Transform(ABC):
    """Abstract base class for a point cloud spatial transform.

    Subclasses represent a concrete transform parametrization (e.g. a
    rigid 4x4 matrix) and must implement how it is applied to points and
    how it is persisted to and loaded from disk.
    """

    @abstractmethod
    def apply_to_points(self, points: PointCloud) -> PointCloud:
        """Apply the transform to a point cloud.

        Parameters
        ----------
        points : PointCloud
            Point cloud to transform.

        Returns
        -------
        PointCloud
            The transformed point cloud.
        """
        raise NotImplementedError

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """Return the transform's parameters as an ndarray.

        Returns
        -------
        np.ndarray
            Array representation of the transform (e.g. a 4x4 homogeneous
            matrix for a rigid transform), suitable for stacking across
            multiple transforms.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the transform's parameters to disk.

        Parameters
        ----------
        path : Path
            Destination file path.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Transform:
        """Load a transform's parameters from disk.

        Parameters
        ----------
        path : Path
            Source file path.

        Returns
        -------
        Transform
            The loaded transform instance.
        """
        raise NotImplementedError
