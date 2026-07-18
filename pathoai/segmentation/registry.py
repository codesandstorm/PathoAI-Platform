"""
pathoai/segmentation/registry.py
================================
Model architecture registry for semantic segmentation.

Enables decoupled architecture registration, mapping string keys (e.g. 'deeplabv3plus')
to model classes without if-else chain coupling.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.1
"""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)

# Registry dictionary
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator to register a custom neural network architecture class.

    Parameters
    ----------
    name : str
        The unique string identifier for the model.

    Returns
    -------
    Callable
        The decorator function.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        lower_name = name.lower()
        if lower_name in _MODEL_REGISTRY:
            logger.warning("Overwriting model architecture registry key: %s", lower_name)
        _MODEL_REGISTRY[lower_name] = cls
        logger.debug("Registered model architecture: %s -> %s", lower_name, cls.__name__)
        return cls
    return decorator


def get_model_class(name: str) -> Type[nn.Module]:
    """Retrieve a registered model class by its string key.

    Parameters
    ----------
    name : str
        The registration key of the model.

    Returns
    -------
    Type[nn.Module]
        The registered model architecture class.

    Raises
    ------
    ValidationError
        If the model identifier is not registered.
    """
    lower_name = name.lower()
    if lower_name not in _MODEL_REGISTRY:
        raise ValidationError(
            f"Model architecture '{name}' not found in registry. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[lower_name]


def list_registered_models() -> list[str]:
    """Return all registered model key names."""
    return list(_MODEL_REGISTRY.keys())
