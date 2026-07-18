"""
tests/unit/segmentation/test_export.py
======================================
Unit tests for the model Export compilation utility.

Verifies:
- Exporting to PyTorch weights format
- Exporting to TorchScript compiled formats (JIT trace)
- Exporting to ONNX graph format

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pathoai.segmentation.export import export_model


class SimpleConvModel(nn.Module):
    """Simple conv model with parameters for tracing tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestModelExport:
    """Verifies that model compilation and exports write target files."""

    def test_export_formats(self, tmp_path: Path):
        model = SimpleConvModel()

        path_pt = tmp_path / "model.pt"
        path_ts = tmp_path / "model_ts.pt"
        path_onnx = tmp_path / "model.onnx"

        # 1. PyTorch weights
        p1 = export_model(model, path_pt, export_format="pytorch")
        assert p1.is_file()

        # 2. TorchScript JIT compile
        p2 = export_model(
            model,
            path_ts,
            export_format="torchscript",
            input_shape=(1, 3, 32, 32),
        )
        assert p2.is_file()

        # 3. ONNX Graph export
        try:
            p3 = export_model(
                model,
                path_onnx,
                export_format="onnx",
                input_shape=(1, 3, 32, 32),
            )
            assert p3.is_file()
        except RuntimeError as exc:
            # Check if it was because of missing onnx/onnxscript packages
            if "onnx" in str(exc).lower():
                pytest.skip("ONNX or onnxscript packages not available in environment.")
            else:
                raise exc
