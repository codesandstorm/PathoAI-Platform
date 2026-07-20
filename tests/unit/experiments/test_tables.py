"""
tests/unit/experiments/test_tables.py
======================================
Unit tests for PublicationTableGenerator.

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
from pathoai.experiments.tables import PublicationTableGenerator


class TestPublicationTableGenerator:
    """Test publication markdown table rendering."""

    def test_generate_publication_tables(self):
        """Test Markdown table creation."""
        res = ValidationResult(
            experiment_name="exp_test",
            dataset_name="TIGER_Val",
            slide_count=5,
            segmentation_metrics=SegmentationMetrics(
                dice=0.88, iou=0.79, precision=0.90, recall=0.86, specificity=0.95, pixel_accuracy=0.92, f1=0.88
            ),
            detection_metrics=DetectionMetrics(
                precision=0.85, recall=0.82, f1=0.83, ap50=0.84, ap75=0.75, map5095=0.80, tp=100, fp=15, fn=20
            ),
            scoring_metrics=ScoringMetrics(
                mae=4.5, rmse=5.8, pearson_r=0.92, pearson_pvalue=1e-5, spearman_r=0.91, spearman_pvalue=1e-5,
                r2=0.85, icc=0.90, bland_altman_bias=0.8, bland_altman_lower_limit=-8.2, bland_altman_upper_limit=9.8
            ),
            statistical_analysis=StatisticalAnalysis(confidence_intervals={}, bootstrap_results={}, p_values={}, effect_sizes={}),
            benchmark_results=BenchmarkResults(baseline_name="base", target_metrics={}, baseline_metrics={}, percentage_improvements={}),
            error_analysis=ErrorAnalysis(false_positives_count=15, false_negatives_count=20, outlier_slides=[], failure_modes={}),
        )

        table_gen = PublicationTableGenerator()
        t1 = table_gen.generate_table_1_segmentation(res)
        t2 = table_gen.generate_table_2_detection(res)
        t3 = table_gen.generate_table_3_agreement(res)

        assert "Table 1:" in t1
        assert "0.88" in t1
        assert "Table 2:" in t2
        assert "Table 3:" in t3
        assert "0.90" in t3
