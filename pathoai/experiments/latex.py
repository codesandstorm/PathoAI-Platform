"""
pathoai/experiments/latex.py
=============================
LaTeX Publication Exporter.

Exports publication-ready LaTeX table code for submission to Nature Medicine, MedIA, or IEEE TMI.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5.6
"""

from __future__ import annotations

from pathoai.core.types import ValidationResult


class LaTeXExporter:
    """Exports LaTeX table strings for manuscript inclusion."""

    def export_latex_table_3_agreement(self, result: ValidationResult) -> str:
        """Exports Table 3 as LaTeX code."""
        s = result.scoring_metrics
        latex = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Clinical sTIL Scoring Agreement against Pathologist Ground Truth.}\n"
            "\\label{tab:clinical_agreement}\n"
            "\\begin{tabular}{lc}\n"
            "\\hline\n"
            "\\textbf{Metric} & \\textbf{Value} \\\\\n"
            "\\hline\n"
            f"Intraclass Correlation (ICC) & \\textbf{{{s.icc:.4f}}} \\\\\n"
            f"Pearson Correlation ($r$) & \\textbf{{{s.pearson_r:.4f}}} \\\\\n"
            f"Spearman Correlation ($\\rho$) & \\textbf{{{s.spearman_r:.4f}}} \\\\\n"
            f"Coefficient of Determination ($R^2$) & {s.r2:.4f} \\\\\n"
            f"Mean Absolute Error (MAE) & {s.mae:.2f}\\% \\\\\n"
            f"Root Mean Squared Error (RMSE) & {s.rmse:.2f}\\% \\\\\n"
            f"Bland--Altman Bias & {s.bland_altman_bias:.2f}\\% \\\\\n"
            f"Bland--Altman 95\\% LoA & [{s.bland_altman_lower_limit:.2f}\\%, {s.bland_altman_upper_limit:.2f}\\%] \\\\\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        return latex
