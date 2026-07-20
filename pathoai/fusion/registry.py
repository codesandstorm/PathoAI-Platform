"""
pathoai/fusion/registry.py
===========================
Spatial Fusion Operations Registry.

Enables decoupled registration of spatial fusion routines and algorithms.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.1
"""

from __future__ import annotations

from typing import Callable, Dict, List

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)

_FUSION_OP_REGISTRY: Dict[str, Callable] = {}


def register_fusion_op(name: str) -> Callable[[Callable], Callable]:
    """Decorator to register a custom spatial fusion operation.

    Parameters
    ----------
    name : str
        Unique identifier for the fusion operation.

    Returns
    -------
    Callable
        The decorator function.
    """
    def decorator(fn: Callable) -> Callable:
        lower_name = name.lower()
        if lower_name in _FUSION_OP_REGISTRY:
            logger.warning("Overwriting fusion op registry key: %s", lower_name)
        _FUSION_OP_REGISTRY[lower_name] = fn
        logger.debug("Registered spatial fusion operation: %s -> %s", lower_name, fn.__name__)
        return fn
    return decorator


def get_fusion_op(name: str) -> Callable:
    """Retrieve a registered spatial fusion operation by key name."""
    lower_name = name.lower()
    if lower_name not in _FUSION_OP_REGISTRY:
        raise ValidationError(
            f"Spatial fusion operation '{name}' not found in registry. "
            f"Available ops: {list(_FUSION_OP_REGISTRY.keys())}"
        )
    return _FUSION_OP_REGISTRY[lower_name]


def list_registered_fusion_ops() -> List[str]:
    """Return all registered spatial fusion operation keys."""
    return list(_FUSION_OP_REGISTRY.keys())
