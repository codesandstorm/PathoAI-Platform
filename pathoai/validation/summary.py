"""
pathoai/validation/summary.py
==============================
Validation Summary Generator.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.17
"""

from __future__ import annotations

from typing import Any, Dict

from pathoai.core.types import ValidationResult


def generate_validation_summary(result: ValidationResult) -> Dict[str, Any]:
    """Compiles structured executive summary dictionary."""
    md = (
        f"# Validation Summary: {result.experiment_name}\n\n"
        f"- **Dataset**: `{result.dataset_name}` ({result.slide_count} slides)\n"
        f"- **Segmentation Dice**: {result.segmentation_metrics.dice:.4f}\n"
        f"- **Detection F1**: {result.detection_metrics.f1:.4f}\n"
        f"- **sTIL ICC Agreement**: {result.scoring_metrics.icc:.4f}\n"
        f"- **sTIL MAE**: {result.scoring_metrics.mae:.2f}%\n"
    )

    return {
        "experiment_name": result.experiment_name,
        "dataset_name": result.dataset_name,
        "slide_count": result.slide_count,
        "markdown_summary": md,
    }
