"""
pathoai/segmentation/export.py
==============================
Model Export and Compilation Utility.

Exports trained segmentation models to PyTorch checkpoints, compiled TorchScript,
or optimized ONNX graph formats for production environments.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.9
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)


def export_model(
    model: nn.Module,
    output_path: str | Path,
    export_format: str = "onnx",
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
) -> Path:
    """Compile and export the model to a deployment format.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to export (or wrapped SegmentationModel).
    output_path : str | Path
        Destination file path.
    export_format : str
        Target format: 'onnx', 'torchscript', or 'pytorch'.
    input_shape : Tuple[int, int, int, int]
        Input shape tuple (B, C, H, W) for graph tracing.

    Returns
    -------
    Path
        Path to the written export file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = export_format.lower()

    # If it is wrapped in SegmentationModel, extract internal PyTorch module
    from pathoai.segmentation.model import SegmentationModel
    raw_model = model.model if isinstance(model, SegmentationModel) else model

    # Ensure model is in eval mode on CPU for compilation/tracing
    raw_model.eval()
    device = next(raw_model.parameters()).device
    dummy_input = torch.zeros(input_shape, dtype=torch.float32, device=device)

    logger.info(
        "Exporting model",
        extra={"format": fmt, "path": str(out_path), "input_shape": input_shape},
    )

    if fmt == "pytorch":
        # Pure weights checkpoint
        torch.save(raw_model.state_dict(), out_path)

    elif fmt == "torchscript":
        try:
            # We use tracing since semantic segmentation has a static feedforward structure
            traced = torch.jit.trace(raw_model, dummy_input)
            traced.save(out_path)
        except Exception as exc:
            logger.error("Failed to trace model with TorchScript: %s", exc)
            raise RuntimeError(f"TorchScript tracing failed: {exc}") from exc

    elif fmt == "onnx":
        try:
            import onnx  # noqa: F401
            import onnxscript  # noqa: F401
        except ImportError as exc:
            logger.error("ONNX/onnxscript is not installed or functional: %s", exc)
            raise RuntimeError(f"ONNX export requires 'onnx' and 'onnxscript' packages: {exc}") from exc

        try:
            torch.onnx.export(
                raw_model,
                dummy_input,
                out_path,
                export_params=True,
                opset_version=14,  # standard modern opset
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size", 2: "height", 3: "width"},
                },
            )
        except Exception as exc:
            logger.error("Failed to export model to ONNX: %s", exc)
            raise RuntimeError(f"ONNX export failed: {exc}") from exc
    else:
        raise ValidationError(
            f"Unsupported export format: '{export_format}'. "
            f"Supported options: 'onnx', 'torchscript', 'pytorch'"
        )

    logger.info("Model exported successfully to %s", out_path)
    return out_path
