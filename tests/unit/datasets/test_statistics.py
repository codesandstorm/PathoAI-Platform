"""
tests/unit/datasets/test_statistics.py
======================================
Unit tests for dataset statistics and weights.

Tests cover:
- compute_class_frequencies aggregation from manifest entries
- calculate_class_loss_weights (inverse frequency, smooth, normalization)
- generate_dataset_statistics_report structure and calculations

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

import pytest

from pathoai.datasets.statistics import (
    calculate_class_loss_weights,
    compute_class_frequencies,
    generate_dataset_statistics_report,
)


class TestDatasetStatistics:
    """Verifies stats aggregation and report formatting."""

    def test_compute_class_frequencies(self):
        entries = [
            {"class_distribution": {"0": 100, "1": 50}},
            {"class_distribution": {"0": 200, "2": 80}},
            {"class_distribution": {"1": 50, "5": 20}},
        ]
        freqs = compute_class_frequencies(entries, n_classes=6)

        assert freqs[0] == 300
        assert freqs[1] == 100
        assert freqs[2] == 80
        assert freqs[3] == 0
        assert freqs[4] == 0
        assert freqs[5] == 20

    def test_calculate_weights_inverse_frequency(self):
        class_counts = {0: 100, 1: 50, 2: 100}  # total 250, C = 3 (non-zero classes)
        # For non-zero classes, mean normalized weight should result in average 1.0
        weights = calculate_class_loss_weights(
            class_counts,
            method="inverse_frequency",
            n_classes=3,
        )

        assert len(weights) == 3
        # Class 1 is half as frequent as Class 0 -> weight should be twice as large
        assert abs(weights[1] - 2 * weights[0]) < 1e-4
        # Normalized mean weight of non-zero elements is 1.0
        assert abs(sum(w for w in weights if w > 0) / 3 - 1.0) < 1e-4

    def test_calculate_weights_smooth_inverse_frequency(self):
        class_counts = {0: 100, 1: 200}
        weights = calculate_class_loss_weights(
            class_counts,
            method="smooth_inverse_frequency",
            n_classes=2,
        )
        assert len(weights) == 2
        # Class 0 is less frequent -> should have larger weight
        assert weights[0] > weights[1]

    def test_calculate_weights_fallback_on_zero_pixels(self):
        class_counts = {0: 0, 1: 0}
        weights = calculate_class_loss_weights(class_counts, n_classes=2)
        assert weights == [1.0, 1.0]

    def test_generate_statistics_report(self):
        train = [
            {"tissue_coverage": 0.80, "class_distribution": {"0": 100}},
            {"tissue_coverage": 0.90, "class_distribution": {"0": 200, "1": 100}},
        ]
        val = [{"tissue_coverage": 0.50, "class_distribution": {"0": 50}}]
        test = []

        report = generate_dataset_statistics_report(train, val, test, n_classes=2)

        # Structure checks
        assert report["summary"]["total_patches"] == 3
        assert report["summary"]["split_counts"]["train"] == 2
        assert report["summary"]["split_counts"]["val"] == 1
        assert report["summary"]["split_counts"]["test"] == 0

        # Profile checks
        assert report["splits_profile"]["train"]["avg_tissue_coverage"] == 0.85
        assert report["splits_profile"]["train"]["class_frequencies"]["0"]["pixel_count"] == 300
