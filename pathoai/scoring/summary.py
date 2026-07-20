"""
pathoai/scoring/summary.py
==========================
sTIL Scoring Summary Generator.

Compiles text and markdown summary statistics of sTIL scoring passes.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.13
"""

from __future__ import annotations

from typing import Any, Dict

from pathoai.core.types import STILScore


def generate_scoring_summary(stil_score: STILScore) -> Dict[str, Any]:
    """Generates structured summary dictionary from STILScore.

    Parameters
    ----------
    stil_score : STILScore
        sTIL scoring DTO.

    Returns
    -------
    Dict[str, Any]
        Structured summary report.
    """
    md = (
        f"# sTIL Scoring Summary Report\n\n"
        f"- **Slide ID**: `{stil_score.slide_id}`\n"
        f"- **sTIL Score %**: **{stil_score.score_percent:.2f}%**\n"
        f"- **Clinical Category**: **{stil_score.clinical_category}**\n"
        f"- **95% Confidence Interval**: [{stil_score.confidence_interval[0]:.2f}%, {stil_score.confidence_interval[1]:.2f}%]\n"
        f"- **Stromal Area**: {stil_score.stromal_area_mm2:.3f} mm²\n"
        f"- **Stromal Lymphocytes**: {stil_score.stromal_lymphocytes:,}\n"
        f"- **Lymphocyte Density**: {stil_score.lymphocyte_density:.1f} cells/mm²\n"
    )

    return {
        "slide_id": stil_score.slide_id,
        "score_percent": stil_score.score_percent,
        "clinical_category": stil_score.clinical_category,
        "confidence_interval": stil_score.confidence_interval,
        "markdown_summary": md,
    }
