"""Abstract base class for point cloud registration evaluation metrics.

This module defines the common interface that every metric implementation
must follow, so that different metrics can be computed interchangeably
during evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """Abstract base class for registration evaluation metrics.

    Subclasses must implement :meth:`compute`. Since different metrics
    operate on different kinds of inputs (e.g. point clouds, transforms,
    correspondences), `compute` accepts arbitrary keyword arguments and
    each subclass documents and validates the keys it expects.
    """

    @abstractmethod
    def compute(self, **inputs: Any) -> dict[str, float]:
        """Compute the metric value(s) from the given inputs.

        Parameters
        ----------
        **inputs : Any
            Subclass-specific named inputs required to compute the metric
            (e.g. `source`, `target`, `transform_pred`, `transform_gt`).

        Returns
        -------
        dict[str, float]
            Mapping from metric name to its computed value.
        """
        raise NotImplementedError
