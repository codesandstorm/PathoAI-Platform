"""
tests/unit/training/test_visualization.py
=========================================
Unit tests for the Visualization Engine.

Verifies:
- plot_training_curves output images
- plot_confusion_matrix layout and plots
- generate_prediction_gallery_row visual strips

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.7
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathoai.training.visualization.confusion import plot_confusion_matrix
from pathoai.training.visualization.curves import plot_training_curves, MATPLOTLIB_AVAILABLE
from pathoai.training.visualization.overlays import generate_prediction_gallery_row


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib backend_agg is blocked or unavailable")
class TestVisualization:
    """Verifies that all Matplotlib visualization routines write correct files."""

    def test_curves_generation(self, tmp_path: Path):
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "mean_dice": 0.70, "mean_iou": 0.60, "learning_rate": 0.01, "elapsed_time": 10.0},
            {"epoch": 2, "train_loss": 0.3, "val_loss": 0.2, "mean_dice": 0.80, "mean_iou": 0.70, "learning_rate": 0.01, "elapsed_time": 12.0},
        ]

        plot_training_curves(history, tmp_path)

        assert (tmp_path / "loss.png").is_file()
        assert (tmp_path / "dice.png").is_file()
        assert (tmp_path / "iou.png").is_file()
        assert (tmp_path / "lr.png").is_file()
        assert (tmp_path / "epoch_time.png").is_file()

    def test_confusion_matrix_generation(self, tmp_path: Path):
        matrix = [[5, 1], [2, 8]]
        class_names = ["Stroma", "Tumor"]
        out_raw = tmp_path / "raw.png"
        out_norm = tmp_path / "norm.png"

        plot_confusion_matrix(matrix, class_names, out_raw, normalize=False)
        plot_confusion_matrix(matrix, class_names, out_norm, normalize=True)

        assert out_raw.is_file()
        assert out_norm.is_file()

    def test_prediction_gallery_generation(self, tmp_path: Path):
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.random.randint(0, 4, (64, 64), dtype=np.uint8)
        pred = np.random.randint(0, 4, (64, 64), dtype=np.uint8)
        out_path = tmp_path / "gallery.png"

        generate_prediction_gallery_row(image, mask, pred, out_path)

        assert out_path.is_file()
