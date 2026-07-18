"""
tests/unit/segmentation/test_summary.py
======================================
Unit tests for the model summary report generator.

Verifies:
- generate_model_summary writes Markdown and text summaries
- Verification of content inside generated summary files

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pathoai.segmentation.model import SegmentationModel
from pathoai.segmentation.summary import generate_model_summary


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestModelSummary:
    """Verifies that model summaries are computed and saved to disk."""

    def test_summary_generation(self, tmp_path: Path):
        raw = SimpleModel()
        model = SegmentationModel(raw)

        summary_data = generate_model_summary(
            model=model,
            output_dir=tmp_path,
            input_shape=(1, 5),
        )

        # Check returned dictionary properties
        assert summary_data["architecture_wrapper"] == "SegmentationModel"
        assert summary_data["output_shape"] == (1, 2)
        assert summary_data["estimated_size_mb"] > 0.0

        # Check files written
        md_file = tmp_path / "Model_Summary.md"
        txt_file = tmp_path / "Model_Summary.txt"
        assert md_file.is_file()
        assert txt_file.is_file()

        content = md_file.read_text(encoding="utf-8")
        assert "PathoAI Model Architecture Summary" in content
        assert "Parameter Profile" in content
        assert "Shape Verifications" in content
