"""Point-based registration evaluation metrics.

This module implements metrics derived from comparing a predicted rigid
transform against its ground-truth counterpart, such as the Relative
Rotation Error (RRE) and Relative Translation Error (RTE).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from registration import Transform

from .base import Metric

logger = logging.getLogger(__name__)


class PointBasedMetric(Metric):
    """Base class for metrics comparing predicted and ground-truth transforms.

    Subclasses use the stored thresholds to derive their specific metric
    (e.g. success rate) from the RRE/RTE computed via the private
    `_compute_rre` and `_compute_rte` static helpers.
    """

    def __init__(self, rte_threshold: float, rre_threshold: float) -> None:
        """Store the thresholds used by this metric family.

        Parameters
        ----------
        rte_threshold : float
            Maximum RTE for a registration to be considered successful.
        rre_threshold : float
            Maximum RRE (in radians) for a registration to be considered
            successful.
        """
        self.rte_threshold = rte_threshold
        self.rre_threshold = rre_threshold

    def compute(self, **inputs: Any) -> dict[str, float]:
        """Compute the RRE and RTE between a predicted and ground-truth transform.

        Parameters
        ----------
        **inputs : Any
            Must contain `transform_gt` and `transform_pred`, both 4x4
            homogeneous transforms as `np.ndarray`.

        Returns
        -------
        dict[str, float]
            Mapping with keys `"rre"` and `"rte"`.
        """
        transform_gt = inputs["transform_gt"]
        transform_pred = inputs["transform_pred"]

        return {
            "rre": self._compute_rre(transform_gt, transform_pred),
            "rte": self._compute_rte(transform_gt, transform_pred),
        }

    @staticmethod
    def _compute_rre(transform_gt: Transform, transform_pred: Transform) -> float:
        """Compute the Relative Rotation Error (RRE) between two transforms.

        Parameters
        ----------
        transform_gt : np.ndarray
            Ground-truth 4x4 homogeneous transform.
        transform_pred : np.ndarray
            Predicted 4x4 homogeneous transform.

        Returns
        -------
        float
            RRE in degrees, computed as
            ``acos((trace(R_pred^T @ R_gt) - 1) / 2)``.
        """
        rotation_gt = transform_gt[:3, :3]
        rotation_pred = transform_pred[:3, :3]

        x = 0.5 * (np.trace(np.matmul(rotation_pred.T, rotation_gt)) - 1.0)
        x = np.clip(x, -1.0, 1.0)
        x = np.arccos(x)
        rre = 180.0 * x / np.pi
        return float(rre)

    @staticmethod
    def _compute_rte(transform_gt: Transform, transform_pred: Transform) -> float:
        """Compute the Relative Translation Error (RTE) between two transforms.

        Parameters
        ----------
        transform_gt : np.ndarray
            Ground-truth 4x4 homogeneous transform.
        transform_pred : np.ndarray
            Predicted 4x4 homogeneous transform.

        Returns
        -------
        float
            RTE, computed as ``||t_gt - t_pred||_2``.
        """

        translation_gt = transform_gt[:3, 3]
        translation_pred = transform_pred[:3, 3]

        return float(np.linalg.norm(translation_gt - translation_pred))
