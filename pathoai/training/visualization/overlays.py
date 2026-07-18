"""
pathoai/training/visualization/overlays.py
=========================================
Prediction Overlays and Error Map Visualizer.

Generates comparison galleries showing the source image, ground-truth mask,
model prediction, true-vs-prediction overlay, and color-coded error maps.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.7
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import matplotlib.pyplot as plt
    # Trigger DLL loading early to catch AppLocker blocks
    import matplotlib.backends.backend_agg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import numpy as np
import torch

from pathoai.core.logger import get_logger
from pathoai.visualization.colormap import build_matplotlib_colormap


logger = get_logger(__name__)


def generate_prediction_gallery_row(
    image: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    pred: Union[torch.Tensor, np.ndarray],
    output_path: str | Path,
) -> None:
    """Generate and save a 5-panel prediction comparison row:

    Image | Ground Truth | Prediction | Overlap Overlay | Error Map

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        Source RGB image. Shape (3, H, W) or (H, W, 3). Value range [0, 1] or [0, 255].
    mask : torch.Tensor | np.ndarray
        Ground-truth class label mask. Shape (H, W).
    pred : torch.Tensor | np.ndarray
        Predicted class label mask. Shape (H, W).
    output_path : str | Path
        Path where the gallery panel figure will be saved.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning(
            "generate_prediction_gallery_row: Matplotlib is not available or blocked. Skipping gallery generation."
        )
        return

    # 1. Coerce image to (H, W, 3) numpy uint8
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.shape[0] == 3:  # (3, H, W)
        image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.01:
        image = (image * 255.0).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # 2. Coerce masks to (H, W) numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    mask = mask.astype(np.uint8)
    pred = pred.astype(np.uint8)

    H, W = mask.shape

    # 3. Create Overlap Overlay
    # Ground Truth: Green, Prediction: Red, Overlap: Yellow, Background: Black/Image
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    gt_pos = mask > 0
    pred_pos = pred > 0

    overlay[gt_pos & ~pred_pos] = [0, 255, 0]   # False Negatives of positive classes -> Green
    overlay[pred_pos & ~gt_pos] = [255, 0, 0]   # False Positives of positive classes -> Red
    overlay[gt_pos & pred_pos] = [255, 255, 0]   # Correct positives -> Yellow

    # 4. Create Error Map
    # Correct positives: Green, False Positives: Red, False Negatives: Blue, TN: Dark Gray
    error_map = np.zeros((H, W, 3), dtype=np.uint8)
    error_map.fill(50)  # background dark gray

    correct_pos = (pred == mask) & (mask > 0)
    false_pos = (pred > 0) & (pred != mask)
    false_neg = (mask > 0) & (pred != mask)

    error_map[correct_pos] = [0, 255, 0]   # Correct positive -> Green
    error_map[false_pos] = [255, 0, 0]      # FP -> Red
    error_map[false_neg] = [0, 0, 255]      # FN -> Blue

    # 5. Colormap for segmentation masks
    cmap = build_matplotlib_colormap()

    # 6. Render the 5 panels
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    axes[0].imshow(image)
    axes[0].set_title("Source Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=5)
    axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=5)
    axes[2].set_title("Prediction", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (GT=G, Pred=R)", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(error_map)
    axes[4].set_title("Error Map (C=G, FP=R, FN=B)", fontsize=12, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out, dpi=200)
        logger.debug("Saved prediction row gallery to %s", out)
    except Exception as exc:
        logger.error("Failed to save prediction row gallery to %s: %s", out, exc)
    finally:
        plt.close()
