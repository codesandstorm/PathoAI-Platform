"""
pathoai/experiments/tables.py
==============================
Publication Tables Generator.

Generates structured Markdown publication tables (Table 1: Segmentation, Table 2: Detection,
Table 3: Clinical Agreement).

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5.5
"""

from __future__ import annotations

from typing import Any, Dict

from pathoai.core.types import ValidationResult


class PublicationTableGenerator:
    """Generates Markdown publication tables."""

    def generate_table_1_segmentation(self, result: ValidationResult) -> str:
        """Generates Table 1: Semantic Segmentation Performance."""
        m = result.segmentation_metrics
        table = (
            f"### Table 1: Semantic Segmentation Performance on {result.dataset_name}\n\n"
            f"| Metric | Score |\n"
            f"| :--- | :---: |\n"
            f"| Dice Similarity Coefficient (DSC) | **{m.dice:.4f}** |\n"
            f"| Intersection over Union (IoU) | **{m.iou:.4f}** |\n"
            f"| Pixel Precision | {m.precision:.4f} |\n"
            f"| Pixel Recall | {m.recall:.4f} |\n"
            f"| Pixel Accuracy | {m.pixel_accuracy:.4f} |\n"
            f"| F1 Score | {m.f1:.4f} |\n"
        )
        return table

    def generate_table_2_detection(self, result: ValidationResult) -> str:
        """Generates Table 2: Cell Detection Performance."""
        d = result.detection_metrics
        table = (
            f"### Table 2: Cell Detection Performance on {result.dataset_name}\n\n"
            f"| Metric | Score |\n"
            f"| :--- | :---: |\n"
            f"| Precision | **{d.precision:.4f}** |\n"
            f"| Recall | **{d.recall:.4f}** |\n"
            f"| F1 Score | **{d.f1:.4f}** |\n"
            f"| AP@50 | {d.ap50:.4f} |\n"
            f"| mAP@50-95 | {d.map5095:.4f} |\n"
            f"| True Positives (TP) | {d.tp:,} |\n"
            f"| False Positives (FP) | {d.fp:,} |\n"
            f"| False Negatives (FN) | {d.fn:,} |\n"
        )
        return table

    def generate_table_3_agreement(self, result: ValidationResult) -> str:
        """Generates Table 3: Clinical sTIL Agreement & Correlation."""
        s = result.scoring_metrics
        table = (
            f"### Table 3: Clinical sTIL Scoring Agreement on {result.dataset_name}\n\n"
            f"| Metric | Value |\n"
            f"| :--- | :---: |\n"
            f"| Intraclass Correlation Coefficient (ICC) | **{s.icc:.4f}** |\n"
            f"| Pearson Correlation Coefficient ($r$) | **{s.pearson_r:.4f}** (p={s.pearson_pvalue:.4e}) |\n"
            f"| Spearman Rank Correlation ($\rho$) | **{s.spearman_r:.4f}** (p={s.spearman_pvalue:.4e}) |\n"
            f"| Coefficient of Determination ($R^2$) | {s.r2:.4f} |\n"
            f"| Mean Absolute Error (MAE) | {s.mae:.2f}% |\n"
            f"| Root Mean Squared Error (RMSE) | {s.rmse:.2f}% |\n"
            f"| Bland–Altman Mean Bias | {s.bland_altman_bias:.2f}% |\n"
            f"| Bland–Altman 95% Limits of Agreement | [{s.bland_altman_lower_limit:.2f}%, {s.bland_altman_upper_limit:.2f}%] |\n"
        )
        return table
