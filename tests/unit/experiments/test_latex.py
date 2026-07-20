"""
tests/unit/experiments/test_latex.py
=====================================
Unit tests for LaTeXExporter.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import (
    BenchmarkResults,
    DetectionMetrics,
    ErrorAnalysis,
    ScoringMetrics,
    SegmentationMetrics,
    StatisticalAnalysis,
    ValidationResult,
)
from pathoai.experiments.latex import LaTeXExporter


class TestLaTeXExporter:
    """Test LaTeX code rendering."""

    def test_export_latex_table(self):
        """Test LaTeX table string export."""
        res = ValidationResult(
            experiment_name="exp_test",
            dataset_name="TIGER_Val",
            slide_count=5,
            segmentation_metrics=SegmentationMetrics(0.88, 0.79, 0.90, 0.86, 0.95, 0.92, 0.88),
            detection_metrics=DetectionMetrics(0.85, 0.82, 0.83, 0.84, 0.75, 0.80, 100, 15, 20),
            scoring_metrics=ScoringMetrics(4.5, 5.8, 0.92, 1e-5, 0.91, 1e-5, 0.85, 0.90, 0.8, -8.2, 9.8),
            statistical_analysis=StatisticalAnalysis({}, {}, {}, {}),
            benchmark_results=BenchmarkResults("base", {}, {}, {}),
            error_analysis=ErrorAnalysis(15, 20, [], {}),
        )

        exporter = LaTeXExporter()
        code = exporter.export_latex_table_3_agreement(res)

        assert "\\begin{table}" in code
        assert "\\caption{" in code
        assert "\\end{table}" in code
