"""
pathoai/scoring/factory.py
==========================
Clinical Scorer Factory.

Instantiates clinical scoring engine components from configuration objects.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.2
"""

from __future__ import annotations

from typing import Any, Dict

from pathoai.core.logger import get_logger
from pathoai.scoring.registry import get_scorer_class

# Import scorer architectures trigger
import pathoai.scoring.scorer  # noqa: F401

logger = get_logger(__name__)


def create_scorer(config: Any = None) -> Any:
    """Create a clinical scorer instance from configuration.

    Parameters
    ----------
    config : Any
        Global configuration object or dictionary.

    Returns
    -------
    Any
        Instantiated clinical scorer.
    """
    if hasattr(config, "scoring"):
        scoring_cfg = config.scoring
    elif isinstance(config, dict) and "scoring" in config:
        scoring_cfg = config["scoring"]
    else:
        scoring_cfg = config or {}

    scorer_key = getattr(scoring_cfg, "algorithm", "tiger_working_group") if not isinstance(scoring_cfg, dict) else scoring_cfg.get("algorithm", "tiger_working_group")
    lymphocyte_diameter_um = getattr(scoring_cfg, "lymphocyte_diameter_um", 10.0) if not isinstance(scoring_cfg, dict) else scoring_cfg.get("lymphocyte_diameter_um", 10.0)

    logger.info("Creating clinical scorer", extra={"algorithm": scorer_key, "lymphocyte_diameter_um": lymphocyte_diameter_um})

    scorer_cls = get_scorer_class(scorer_key)
    return scorer_cls(lymphocyte_diameter_um=lymphocyte_diameter_um)
