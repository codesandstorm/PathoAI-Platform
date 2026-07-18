"""
pathoai/segmentation/summary.py
==============================
Model Summary Report Generator.

Compiles parameter statistics, estimated memory footprints, backbone specs,
and shape assertions into Markdown and text formats for documentation.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.7
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from pathoai.core.logger import get_logger
from pathoai.segmentation.utils import estimate_model_size_mb, verify_output_shape

logger = get_logger(__name__)


def generate_model_summary(
    model: Any,  # SegmentationModel wrapper
    output_dir: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
) -> Dict[str, Any]:
    """Compile model architecture properties and write summary logs to disk.

    Parameters
    ----------
    model : SegmentationModel
        The wrapped SegmentationModel wrapper.
    output_dir : str | Path
        Target save directory.
    input_shape : tuple[int, int, int, int]
        Input shape tuple for validation.

    Returns
    -------
    Dict[str, Any]
        Aggregated summary properties.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather metrics
    param_counts = model.count_parameters()
    model_size_mb = estimate_model_size_mb(model)

    try:
        out_shape = verify_output_shape(model, input_shape)
    except Exception as exc:
        logger.warning("Could not compute dummy forward pass output shape: %s", exc)
        out_shape = ("Unknown", "Unknown", "Unknown", "Unknown")

    encoder = model.get_encoder()
    encoder_name = encoder.__class__.__name__ if encoder else "None / Direct CNN"

    summary_data = {
        "architecture_wrapper": model.__class__.__name__,
        "inner_class": model.model.__class__.__name__,
        "encoder_backbone": encoder_name,
        "parameters": param_counts,
        "estimated_size_mb": model_size_mb,
        "input_shape": input_shape,
        "output_shape": out_shape,
    }

    # 2. Build Markdown Text
    md = []
    md.append("# PathoAI Model Architecture Summary\n")
    md.append("## 🏗️ Core Specification")
    md.append(f"- **Wrapper Class:** `{summary_data['architecture_wrapper']}`")
    md.append(f"- **Internal Architecture:** `{summary_data['inner_class']}`")
    md.append(f"- **Encoder Backbone:** `{summary_data['encoder_backbone']}`\n")

    md.append("## 📊 Parameter Profile")
    md.append(f"- **Total Parameters:** {param_counts['total']:,}")
    md.append(f"- **Trainable Parameters:** {param_counts['trainable']:,}")
    md.append(f"- **Frozen Parameters:** {param_counts['non_trainable']:,}")
    md.append(f"- **Estimated Model Memory Footprint:** {summary_data['estimated_size_mb']:.4f} MB\n")

    md.append("## 📐 Shape Verifications")
    md.append(f"- **Sample Input Shape:** `{input_shape}`")
    md.append(f"- **Asserted Output Shape:** `{out_shape}`\n")

    md_text = "\n".join(md)

    # 3. Save Markdown and Text files
    md_path = out_dir / "Model_Summary.md"
    txt_path = out_dir / "Model_Summary.txt"

    try:
        md_path.write_text(md_text, encoding="utf-8")
        txt_path.write_text(md_text, encoding="utf-8")
        logger.info("Saved model summary report to %s", md_path)
    except Exception as exc:
        logger.error("Failed to write model summary reports: %s", exc)

    return summary_data
