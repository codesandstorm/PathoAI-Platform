"""
pathoai/detection/summary.py
============================
Detector Summary Report Generator.

Formats detection model parameter counts, layer details, and configurations into
structured Markdown and text reports.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.15
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from pathoai.detection.model import DetectionModel


def generate_detector_summary(
    model: DetectionModel, config: Any = None
) -> Dict[str, Any]:
    """Generates structured summary details of a detection model.

    Parameters
    ----------
    model : DetectionModel
        Target detection model wrapper.
    config : Any
        Optional config object.

    Returns
    -------
    Dict[str, Any]
        Dictionary of summary statistics.
    """
    py_model = model.model
    total_params = sum(p.numel() for p in py_model.parameters())
    trainable_params = sum(p.numel() for p in py_model.parameters() if p.requires_grad)

    arch_name = py_model.__class__.__name__

    summary_text = (
        f"# Object Detector Summary: {arch_name}\n\n"
        f"- **Architecture Class**: `{arch_name}`\n"
        f"- **Device**: `{model.device}`\n"
        f"- **Total Parameters**: {total_params:,}\n"
        f"- **Trainable Parameters**: {trainable_params:,}\n"
    )

    return {
        "architecture_name": arch_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(model.device),
        "markdown_summary": summary_text,
    }
