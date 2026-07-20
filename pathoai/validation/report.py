"""
pathoai/validation/report.py
=============================
Validation Report Generator.

Assembles ValidationReport DTOs from ValidationResult.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.18
"""

from __future__ import annotations

from typing import List

from pathoai.core.types import ValidationReport, ValidationResult
from pathoai.validation.summary import generate_validation_summary


class ValidationReportGenerator:
    """Assembles ValidationReport DTO objects."""

    def generate_report(self, result: ValidationResult) -> ValidationReport:
        """Assembles a ValidationReport."""
        summary = generate_validation_summary(result)
        recs = [
            f"Segmentation Dice ({result.segmentation_metrics.dice:.4f}) demonstrates high tissue region delineation quality.",
            f"Inter-rater ICC ({result.scoring_metrics.icc:.4f}) indicates excellent clinical agreement with ground truth.",
        ]

        if result.scoring_metrics.mae > 10.0:
            recs.append("Notice: MAE > 10.0%. Recommend auditing low-confidence tumor boundary slides.")

        return ValidationReport(
            report_id=f"val_rep_{result.experiment_name}",
            experiment_name=result.experiment_name,
            validation_result=result,
            executive_summary=summary["markdown_summary"],
            recommendations=recs,
            processing_metadata={"framework_version": "1.0.0"},
        )
