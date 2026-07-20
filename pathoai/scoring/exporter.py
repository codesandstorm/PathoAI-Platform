"""
pathoai/scoring/exporter.py
===========================
Clinical Score Exporter Engine.

Exports STILScore and ClinicalReport objects to standard JSON, CSV, and Markdown files.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.11
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from pathoai.core.types import ClinicalReport, STILScore


def export_stil_score_to_json(stil_score: STILScore, output_path: Union[str, Path]) -> None:
    """Exports STILScore object to JSON format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "slide_id": stil_score.slide_id,
        "score_percent": stil_score.score_percent,
        "stromal_area_mm2": stil_score.stromal_area_mm2,
        "stromal_lymphocytes": stil_score.stromal_lymphocytes,
        "lymphocyte_density": stil_score.lymphocyte_density,
        "confidence_interval": list(stil_score.confidence_interval),
        "confidence_level": stil_score.confidence_level,
        "clinical_category": stil_score.clinical_category,
        "explanation": stil_score.explanation,
        "metadata": stil_score.metadata,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_clinical_report_to_markdown(report: ClinicalReport, output_path: Union[str, Path]) -> None:
    """Exports ClinicalReport object to Markdown report format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    s = report.stil_score
    md = (
        f"# Clinical sTIL Evaluation Report\n\n"
        f"**Slide ID**: `{report.slide_id}`\n\n"
        f"## 📊 Scoring Summary\n"
        f"- **sTIL Score**: **{s.score_percent:.2f}%**\n"
        f"- **95% Confidence Interval**: [{s.confidence_interval[0]:.2f}%, {s.confidence_interval[1]:.2f}%]\n"
        f"- **Clinical Category**: **{s.clinical_category}**\n"
        f"- **Stromal Lymphocytes**: {s.stromal_lymphocytes:,}\n"
        f"- **Stromal Area**: {s.stromal_area_mm2:.3f} mm²\n"
        f"- **Lymphocyte Density**: {s.lymphocyte_density:.1f} cells/mm²\n\n"
        f"## 📋 Interpretation\n"
        f"{report.interpretation}\n\n"
        f"## 🔍 Rationale & Explainability\n"
        f"{s.explanation}\n\n"
        f"## 💡 Recommendations\n"
    )
    for rec in report.recommendations:
        md += f"- {rec}\n"

    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
