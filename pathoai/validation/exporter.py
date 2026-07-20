"""
pathoai/validation/exporter.py
==============================
Validation Exporter Engine.

Exports ValidationResult and ValidationReport DTOs into standard JSON, CSV, and Markdown.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.16
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from pathoai.core.types import ValidationReport, ValidationResult


def export_validation_result_to_json(result: ValidationResult, output_path: Union[str, Path]) -> None:
    """Exports ValidationResult object to JSON format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "experiment_name": result.experiment_name,
        "dataset_name": result.dataset_name,
        "slide_count": result.slide_count,
        "segmentation_metrics": {
            "dice": result.segmentation_metrics.dice,
            "iou": result.segmentation_metrics.iou,
            "precision": result.segmentation_metrics.precision,
            "recall": result.segmentation_metrics.recall,
            "f1": result.segmentation_metrics.f1,
        },
        "detection_metrics": {
            "precision": result.detection_metrics.precision,
            "recall": result.detection_metrics.recall,
            "f1": result.detection_metrics.f1,
            "ap50": result.detection_metrics.ap50,
            "map5095": result.detection_metrics.map5095,
        },
        "scoring_metrics": {
            "mae": result.scoring_metrics.mae,
            "rmse": result.scoring_metrics.rmse,
            "pearson_r": result.scoring_metrics.pearson_r,
            "spearman_r": result.scoring_metrics.spearman_r,
            "r2": result.scoring_metrics.r2,
            "icc": result.scoring_metrics.icc,
            "bland_altman_bias": result.scoring_metrics.bland_altman_bias,
        },
        "error_analysis": {
            "outlier_slides": result.error_analysis.outlier_slides,
            "failure_modes": result.error_analysis.failure_modes,
        },
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_validation_report_to_markdown(report: ValidationReport, output_path: Union[str, Path]) -> None:
    """Exports ValidationReport to Markdown report format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    res = report.validation_result
    seg = res.segmentation_metrics
    det = res.detection_metrics
    score = res.scoring_metrics

    md = (
        f"# Scientific Validation & Benchmarking Report\n\n"
        f"**Report ID**: `{report.report_id}`  \n"
        f"**Experiment**: `{report.experiment_name}`  \n"
        f"**Dataset**: `{res.dataset_name}` ({res.slide_count} slides)\n\n"
        f"## 📊 Executive Summary\n"
        f"{report.executive_summary}\n\n"
        f"## 🧩 Stage-wise Performance Metrics\n\n"
        f"### 1. Semantic Segmentation\n"
        f"- **Dice Similarity Coefficient**: `{seg.dice:.4f}`\n"
        f"- **IoU**: `{seg.iou:.4f}`\n"
        f"- **Pixel Precision / Recall**: `{seg.precision:.4f}` / `{seg.recall:.4f}`\n\n"
        f"### 2. Cell Detection\n"
        f"- **Precision / Recall / F1**: `{det.precision:.4f}` / `{det.recall:.4f}` / `{det.f1:.4f}`\n"
        f"- **AP@50 / mAP@50-95**: `{det.ap50:.4f}` / `{det.map5095:.4f}`\n\n"
        f"### 3. Clinical sTIL Agreement\n"
        f"- **Intraclass Correlation (ICC)**: `{score.icc:.4f}`\n"
        f"- **Pearson $r$ / Spearman $\\rho$**: `{score.pearson_r:.4f}` / `{score.spearman_r:.4f}`\n"
        f"- **MAE / RMSE**: `{score.mae:.2f}%` / `{score.rmse:.2f}%`\n"
        f"- **Bland–Altman Bias**: `{score.bland_altman_bias:.2f}%` (95% Limits: [{score.bland_altman_lower_limit:.2f}%, {score.bland_altman_upper_limit:.2f}%])\n\n"
        f"## 💡 Recommendations\n"
    )
    for rec in report.recommendations:
        md += f"- {rec}\n"

    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
