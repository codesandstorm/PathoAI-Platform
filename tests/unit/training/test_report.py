"""
tests/unit/training/test_report.py
==================================
Unit tests for the ReportGenerator.

Verifies:
- Markdown report file generation
- Presence of mandatory sections in generated text

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.9
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathoai.training.reports.report_generator import ReportGenerator


class TestReportGenerator:
    """Verifies that report files are generated and formatted correctly."""

    def test_report_generation(self, tmp_path: Path):
        gen = ReportGenerator(output_dir=tmp_path)

        config_dict = {"learning_rate": 0.01, "batch_size": 8}
        history_df = pd.DataFrame([
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "mean_dice": 0.70},
            {"epoch": 2, "train_loss": 0.3, "val_loss": 0.3, "mean_dice": 0.80},
        ])

        best_metrics = {
            "mean_dice": 0.80,
            "mean_iou": 0.70,
            "macro_f1": 0.80,
            "pixel_accuracy": 0.85,
            "dice_per_class": [0.85, 0.75],
            "iou_per_class": [0.75, 0.65],
            "support_per_class": [1000, 2000],
        }

        dataset_summary = {
            "train": {"n_patches": 100, "avg_tissue_coverage": 0.85},
            "val": {"n_patches": 20, "avg_tissue_coverage": 0.80},
        }

        report_path = gen.generate_report(
            experiment_name="test_experiment",
            config_dict=config_dict,
            history_df=history_df,
            best_epoch_metrics=best_metrics,
            best_epoch=2,
            elapsed_time=120.0,
            dataset_summary=dataset_summary,
        )

        assert report_path.is_file()
        content = report_path.read_text(encoding="utf-8")

        # Verify key markdown headers and content
        assert "Experiment Training Report" in content
        assert "Executive Summary" in content
        assert "Environment & Reproducibility" in content
        assert "Dataset Profile" in content
        assert "Class-specific Performance" in content
        assert "Learning Curves" in content
        assert "Confusion Matrix" in content
        assert "Prediction Visualizations" in content
