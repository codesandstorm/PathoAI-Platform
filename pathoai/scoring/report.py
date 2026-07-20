"""
pathoai/scoring/report.py
=========================
Clinical Report Generator Engine.

Assembles comprehensive ClinicalReport DTO instances with clinical interpretations,
explainable rationales, and recommendations.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.14
"""

from __future__ import annotations

from typing import List, Optional

from pathoai.core.types import ClinicalReport, STILScore
from pathoai.scoring.clinical_rules import ClinicalRules


class ReportGenerator:
    """Assembles ClinicalReport objects."""

    def __init__(self, rules: Optional[ClinicalRules] = None) -> None:
        self.rules = rules or ClinicalRules()

    def generate_report(self, stil_score: STILScore) -> ClinicalReport:
        """Assembles a ClinicalReport from a STILScore instance.

        Parameters
        ----------
        stil_score : STILScore
            Target sTIL score.

        Returns
        -------
        ClinicalReport
            Structured clinical report container DTO.
        """
        interp = self.rules.get_interpretation(stil_score.score_percent)
        recommendations: List[str] = []

        if stil_score.clinical_category == "Low":
            recommendations.append("Low sTIL score indicates immune-cold tumor stroma. Recommend correlation with additional biomarkers.")
        elif stil_score.clinical_category == "Intermediate":
            recommendations.append("Moderate sTIL score. Recommend monitoring and standard clinical follow-up.")
        else:
            recommendations.append("High sTIL score (>50%) indicates immune-rich tumor stroma, favorable for immunotherapeutic response.")

        ci_w = stil_score.confidence_interval[1] - stil_score.confidence_interval[0]
        if ci_w > 20.0:
            recommendations.append(f"Note: Wide 95% confidence interval ({ci_w:.1f}%). Recommend verifying ROI boundaries.")

        return ClinicalReport(
            slide_id=stil_score.slide_id,
            stil_score=stil_score,
            interpretation=interp,
            summary={
                "score_percent": stil_score.score_percent,
                "category": stil_score.clinical_category,
                "confidence_interval": stil_score.confidence_interval,
            },
            recommendations=recommendations,
            processing_metadata={"engine_version": "0.9.0"},
        )
