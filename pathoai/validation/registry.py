"""
pathoai/validation/registry.py
==============================
Validation Stage Evaluators Registry.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.1
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)

_EVALUATOR_REGISTRY: Dict[str, Type] = {}


def register_evaluator(name: str) -> Callable[[Type], Type]:
    """Decorator to register a custom stage evaluator.

    Parameters
    ----------
    name : str
        Unique string key for the evaluator.

    Returns
    -------
    Callable
        Decorator function.
    """
    def decorator(cls: Type) -> Type:
        lower_name = name.lower()
        if lower_name in _EVALUATOR_REGISTRY:
            logger.warning("Overwriting validation evaluator key: %s", lower_name)
        _EVALUATOR_REGISTRY[lower_name] = cls
        logger.debug("Registered validation evaluator: %s -> %s", lower_name, cls.__name__)
        return cls
    return decorator


def get_evaluator_class(name: str) -> Type:
    """Retrieve a registered evaluator class by name."""
    lower_name = name.lower()
    if lower_name not in _EVALUATOR_REGISTRY:
        raise ValidationError(
            f"Validation evaluator '{name}' not found in registry. "
            f"Available evaluators: {list(_EVALUATOR_REGISTRY.keys())}"
        )
    return _EVALUATOR_REGISTRY[lower_name]


def list_registered_evaluators() -> List[str]:
    """Return all registered evaluator keys."""
    return list(_EVALUATOR_REGISTRY.keys())
