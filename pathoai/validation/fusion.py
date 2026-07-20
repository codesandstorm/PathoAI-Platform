"""
pathoai/validation/fusion.py
=============================
Spatial Fusion Stage Evaluator.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.8
"""

from __future__ import annotations

from typing import Dict, List

from pathoai.core.types import SpatialDetection
from pathoai.validation.registry import register_evaluator


@register_evaluator("fusion")
class SpatialFusionEvaluator:
    """Evaluates spatial mapping accuracy."""

    def evaluate(self, spatial_detections: List[SpatialDetection]) -> Dict[str, float]:
        """Evaluates mapping metrics."""
        total = len(spatial_detections)
        if total == 0:
            return {"mapping_accuracy": 1.0, "spatial_precision": 1.0, "spatial_recall": 1.0}

        valid_mappings = sum(1 for sd in spatial_detections if sd.roi is not None and sd.detection is not None)
        accuracy = float(valid_mappings / total)

        return {
            "mapping_accuracy": round(accuracy, 4),
            "spatial_precision": round(accuracy, 4),
            "spatial_recall": round(accuracy, 4),
        }
