"""
pathoai/scoring/registry.py
===========================
Clinical Scorer Registry.

Enables decoupled registration of clinical scoring algorithm implementations.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.1
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)

_SCORER_REGISTRY: Dict[str, Type] = {}


def register_scorer(name: str) -> Callable[[Type], Type]:
    """Decorator to register a custom clinical scoring engine class.

    Parameters
    ----------
    name : str
        Unique string key for the scoring implementation (e.g. 'tiger_working_group').

    Returns
    -------
    Callable
        The decorator function.
    """
    def decorator(cls: Type) -> Type:
        lower_name = name.lower()
        if lower_name in _SCORER_REGISTRY:
            logger.warning("Overwriting clinical scorer registry key: %s", lower_name)
        _SCORER_REGISTRY[lower_name] = cls
        logger.debug("Registered clinical scorer: %s -> %s", lower_name, cls.__name__)
        return cls
    return decorator


def get_scorer_class(name: str) -> Type:
    """Retrieve a registered clinical scorer class by key name."""
    lower_name = name.lower()
    if lower_name not in _SCORER_REGISTRY:
        raise ValidationError(
            f"Clinical scorer '{name}' not found in registry. "
            f"Available scorers: {list(_SCORER_REGISTRY.keys())}"
        )
    return _SCORER_REGISTRY[lower_name]


def list_registered_scorers() -> List[str]:
    """Return all registered scorer keys."""
    return list(_SCORER_REGISTRY.keys())
