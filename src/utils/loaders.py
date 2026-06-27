"""Factory functions to instantiate registration methods, metrics, and
datasets by name.

This module keeps the concrete `RegistrationMethod`, `Metric`, and
`Dataset` subclasses out of call sites (e.g. configs and the experiment
pipeline), so that the abstractions defined in `registration.base` and
`metrics.base` do not leak elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import Dataset

from datasets import V2XReal
from metrics import Metric, PointBasedMetric
from registration import RegistrationMethod, KISSMatcher

logger = logging.getLogger(__name__)

# Empty until concrete subclasses are implemented; populate with
# "name": ClassName entries as methods/metrics/datasets are added.
_METHODS: dict[str, type[RegistrationMethod]] = {"kiss_matcher": KISSMatcher}
_METRICS: dict[str, type[Metric]] = {"point_based": PointBasedMetric}
_DATASETS: dict[str, type[Dataset]] = {"v2x_real": V2XReal}

_REQUIRED_TOP_LEVEL_KEYS = (
    "output_path",
    "transforms_path",
    "overlap_threshold",
    "registration_method",
    "dataset",
    "metrics",
)
_REQUIRED_COMPONENT_KEYS = ("name", "params")


class ConfigError(ValueError):
    """Raised when a run configuration is missing required keys."""


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
        raise ValueError(f"Unknown {kind} '{name}'. Available {kind}s: {available}")

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


def _validate_component(component_cfg: Any, label: str) -> None:
    """Check that a method/dataset/metric entry has a `name` and `params` key.

    Parameters
    ----------
    component_cfg : Any
        The entry to validate (e.g. `config["dataset"]`).
    label : str
        Human-readable label for the entry, used in the error message.

    Raises
    ------
    ConfigError
        If `component_cfg` is not a dict, or is missing a `name` or
        `params` key.
    """
    if not isinstance(component_cfg, dict):
        raise ConfigError(f"Malformed configuration: '{label}' must be a mapping.")

    missing = [key for key in _REQUIRED_COMPONENT_KEYS if key not in component_cfg]
    if missing:
        raise ConfigError(
            f"Malformed configuration: '{label}' is missing key(s) {missing}."
        )


def _validate_config(config: dict[str, Any]) -> None:
    """Check that a parsed configuration has all required keys.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed run configuration.

    Raises
    ------
    ConfigError
        If a required top-level key is missing, or if
        `registration_method`, `dataset`, or any entry of `metrics` is
        missing a `name` or `params` key.
    """
    missing = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ConfigError(f"Malformed configuration: missing key(s) {missing}.")

    for component in ("registration_method", "dataset"):
        _validate_component(config[component], component)

    for idx, metric_cfg in enumerate(config["metrics"]):
        _validate_component(metric_cfg, f"metrics[{idx}]")


def load_config(config_path: Path) -> dict[str, Any]:
    """Read, parse, and validate a YAML run configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        The parsed configuration.

    Raises
    ------
    ConfigError
        If the configuration is missing required keys.
    """
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


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
