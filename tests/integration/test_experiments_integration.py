"""
tests/integration/test_experiments_integration.py
===================================================
Integration tests for Milestone 10.5 Experiment Tracking & Reproducibility Framework.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import json

from pathoai.config.experiment_config import ExperimentConfig
from pathoai.core.types import (
    BenchmarkResults,
    DetectionMetrics,
    ErrorAnalysis,
    ScoringMetrics,
    SegmentationMetrics,
    StatisticalAnalysis,
    ValidationReport,
    ValidationResult,
)
from pathoai.experiments.supplementary import SupplementaryPackageGenerator
from pathoai.experiments.tracker import ExperimentTracker


def test_end_to_end_experiments_tracking_integration(tmp_path):
    """Verifies end-to-end experiment logging and supplementary research packaging."""
    cfg = ExperimentConfig(experiment_id="exp_nature_med_001")

    res = ValidationResult(
        experiment_name="exp_nature_med_001",
        dataset_name="TIGER_Grand_Challenge",
        slide_count=10,
        segmentation_metrics=SegmentationMetrics(0.91, 0.83, 0.92, 0.90, 0.96, 0.94, 0.91),
        detection_metrics=DetectionMetrics(0.88, 0.85, 0.86, 0.87, 0.78, 0.82, 500, 40, 60),
        scoring_metrics=ScoringMetrics(3.2, 4.5, 0.94, 1e-6, 0.93, 1e-6, 0.88, 0.92, 0.5, -6.5, 7.5),
        statistical_analysis=StatisticalAnalysis({}, {}, {}, {}),
        benchmark_results=BenchmarkResults("TIGER_Base", {}, {}, {}),
        error_analysis=ErrorAnalysis(40, 60, ["slide_99"], {}),
    )

    tracker = ExperimentTracker(base_dir=tmp_path / "exps")
    exp_dir = tracker.log_experiment(cfg, res, duration_s=12.5)

    assert (exp_dir / "manifest.json").is_file()
    assert (exp_dir / "metrics.json").is_file()

    with open(exp_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["experiment_id"] == "exp_nature_med_001"
    assert manifest_data["dataset_name"] == "TIGER_Grand_Challenge"

    report = ValidationReport(
        report_id="rep_01",
        experiment_name="exp_nature_med_001",
        validation_result=res,
        executive_summary="Excellent clinical agreement.",
    )

    supp_gen = SupplementaryPackageGenerator()
    supp_dir = supp_gen.generate_supplementary_package(report, tmp_path / "supp_pkg")

    assert (supp_dir / "manifest.json").is_file()
    assert (supp_dir / "publication_tables.md").is_file()
    assert (supp_dir / "table3_agreement.tex").is_file()
