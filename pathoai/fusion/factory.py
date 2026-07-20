"""
pathoai/fusion/factory.py
=========================
Spatial Fusion Factory.

Instantiates spatial fusion components and pipeline instances from configuration objects.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.2
"""

from __future__ import annotations

from typing import Any, Dict

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


def create_fusion_engine(config: Any = None) -> Dict[str, Any]:
    """Creates a dictionary of spatial fusion configuration settings.

    Parameters
    ----------
    config : Any
        Global configuration object or dictionary.

    Returns
    -------
    Dict[str, Any]
        Instantiated configuration settings for FusionPipeline.
    """
    if hasattr(config, "fusion"):
        fusion_cfg = config.fusion
    elif isinstance(config, dict) and "fusion" in config:
        fusion_cfg = config["fusion"]
    else:
        fusion_cfg = config or {}

    grid_size = getattr(fusion_cfg, "grid_size", 1024) if not isinstance(fusion_cfg, dict) else fusion_cfg.get("grid_size", 1024)
    max_distance_um = getattr(fusion_cfg, "max_distance_um", 1000.0) if not isinstance(fusion_cfg, dict) else fusion_cfg.get("max_distance_um", 1000.0)

    logger.info("Created spatial fusion engine configuration", extra={"grid_size": grid_size, "max_distance_um": max_distance_um})

    return {
        "grid_size": grid_size,
        "max_distance_um": max_distance_um,
    }
