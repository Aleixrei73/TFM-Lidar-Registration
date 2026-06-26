"""Factory functions to instantiate registration methods, metrics, and
datasets by name.

This module keeps the concrete `RegistrationMethod`, `Metric`, and
`Dataset` subclasses out of call sites (e.g. configs and the experiment
pipeline), so that the abstractions defined in `registration.base` and
`metrics.base` do not leak elsewhere.
"""

from __future__ import annotations

import logging
from typing import Any

from torch.utils.data import Dataset

from metrics import Metric
from registration import RegistrationMethod

logger = logging.getLogger(__name__)

# Empty until concrete subclasses are implemented; populate with
# "name": ClassName entries as methods/metrics/datasets are added.
_METHODS: dict[str, type[RegistrationMethod]] = {}
_METRICS: dict[str, type[Metric]] = {}
_DATASETS: dict[str, type[Dataset]] = {}


def _load(registry: dict[str, type], name: str, kind: str, **kwargs: Any) -> Any:
    """Look up `name` in `registry` and instantiate it with `kwargs`.

    Parameters
    ----------
    registry : dict[str, type]
        Mapping from registered name to class.
    name : str
        Name of the class to instantiate.
    kind : str
        Human-readable category of `registry` (e.g. "metric"), used to
        build an informative error message.
    **kwargs : Any
        Keyword arguments forwarded to the class constructor.

    Returns
    -------
    Any
        A new instance of the class registered under `name`.

    Raises
    ------
    ValueError
        If `name` is not present in `registry`.
    """
    if name not in registry:
        available = sorted(registry)
        raise ValueError(
            f"Unknown {kind} '{name}'. Available {kind}s: {available}"
        )

    cls = registry[name]
    logger.debug("Instantiating %s '%s' as %s", kind, name, cls.__name__)
    return cls(**kwargs)


def load_method(name: str, **kwargs: Any) -> RegistrationMethod:
    """Instantiate a registration method by name.

    Parameters
    ----------
    name : str
        Name of the registration method, as registered in `_METHODS`.
    **kwargs : Any
        Keyword arguments forwarded to the method's constructor.

    Returns
    -------
    RegistrationMethod
        The instantiated registration method.
    """
    return _load(_METHODS, name, "registration method", **kwargs)


def load_metric(name: str, **kwargs: Any) -> Metric:
    """Instantiate a metric by name.

    Parameters
    ----------
    name : str
        Name of the metric, as registered in `_METRICS`.
    **kwargs : Any
        Keyword arguments forwarded to the metric's constructor.

    Returns
    -------
    Metric
        The instantiated metric.
    """
    return _load(_METRICS, name, "metric", **kwargs)


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    """Instantiate a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset, as registered in `_DATASETS`.
    **kwargs : Any
        Keyword arguments forwarded to the dataset's constructor.

    Returns
    -------
    Dataset
        The instantiated dataset.
    """
    return _load(_DATASETS, name, "dataset", **kwargs)
