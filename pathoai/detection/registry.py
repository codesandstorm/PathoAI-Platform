"""
pathoai/detection/registry.py
==============================
Detector architecture registry for object detection.

Enables decoupled architecture registration, mapping string keys (e.g. 'yolo')
to detector model classes without hardcoding if-else chains.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.1
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type

import torch.nn as nn

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)

# Registry dictionary
_DETECTOR_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_detector(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator to register a custom object detection neural network architecture class.

    Parameters
    ----------
    name : str
        The unique string identifier for the detector (e.g. 'yolo', 'rtdetr', 'cellvit').

    Returns
    -------
    Callable
        The decorator function.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        lower_name = name.lower()
        if lower_name in _DETECTOR_REGISTRY:
            logger.warning("Overwriting detector architecture registry key: %s", lower_name)
        _DETECTOR_REGISTRY[lower_name] = cls
        logger.debug("Registered detector architecture: %s -> %s", lower_name, cls.__name__)
        return cls
    return decorator


def get_detector_class(name: str) -> Type[nn.Module]:
    """Retrieve a registered detector class by its string key.

    Parameters
    ----------
    name : str
        The registration key of the detector architecture.

    Returns
    -------
    Type[nn.Module]
        The registered detector architecture class.

    Raises
    ------
    ValidationError
        If the detector identifier is not registered.
    """
    lower_name = name.lower()
    if lower_name not in _DETECTOR_REGISTRY:
        raise ValidationError(
            f"Detector architecture '{name}' not found in registry. "
            f"Available detectors: {list(_DETECTOR_REGISTRY.keys())}"
        )
    return _DETECTOR_REGISTRY[lower_name]


def list_registered_detectors() -> List[str]:
    """Return all registered detector key names."""
    return list(_DETECTOR_REGISTRY.keys())
