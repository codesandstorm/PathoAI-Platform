"""
pathoai/validation/factory.py
=============================
Validation Evaluator Factory.

Instantiates stage evaluators and validation pipelines from configuration objects.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.2
"""

from __future__ import annotations

from typing import Any, Dict

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


def create_evaluator(stage_name: str, config: Any = None) -> Any:
    """Create a validation evaluator instance for a specific pipeline stage.

    Parameters
    ----------
    stage_name : str
        Stage name (e.g. 'segmentation', 'detection', 'scoring').
    config : Any
        Configuration object or dictionary.

    Returns
    -------
    Any
        Instantiated evaluator.
    """
    logger.info("Creating validation evaluator for stage: %s", stage_name)
    if stage_name.lower() == "segmentation":
        from pathoai.validation.segmentation import SegmentationEvaluator
        return SegmentationEvaluator()
    elif stage_name.lower() == "detection":
        from pathoai.validation.detection import DetectionEvaluator
        return DetectionEvaluator()
    elif stage_name.lower() == "scoring":
        from pathoai.validation.scoring import ClinicalScoringEvaluator
        return ClinicalScoringEvaluator()
    else:
        raise ValueError(f"Unknown validation stage_name: {stage_name}")
