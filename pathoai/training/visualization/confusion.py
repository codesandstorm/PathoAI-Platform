"""
pathoai/training/visualization/confusion.py
===========================================
Confusion Matrix Plotter.

Plots raw and normalized confusion matrices as clean heatmaps.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.7
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Union

try:
    import matplotlib.pyplot as plt
    # Trigger DLL loading early to catch AppLocker blocks
    import matplotlib.backends.backend_agg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import numpy as np

from pathoai.core.logger import get_logger


logger = get_logger(__name__)


def plot_confusion_matrix(
    matrix: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    output_path: str | Path,
    normalize: bool = False,
    cmap: str = "Blues",
) -> None:
    """Plot a confusion matrix heatmap and save to disk.

    Parameters
    ----------
    matrix : List[List[int]] | np.ndarray
        Confusion matrix values (C x C).
    class_names : List[str]
        List of human-readable class names.
    output_path : str | Path
        Path to save the generated image.
    normalize : bool
        If True, normalizes row-wise (row sum to 1.0).
    cmap : str
        Color map.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("plot_confusion_matrix: Matplotlib is not available or blocked. Skipping confusion plot.")
        return

    matrix = np.array(matrix, dtype=np.float64)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    title = "Confusion Matrix"
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)
        title = "Normalized " + title

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Determine thresholds for text colors (contrast check)
    thresh = matrix.max() / 2.0
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        val = matrix[i, j]
        text_val = f"{val:.2f}" if normalize else f"{int(val)}"
        plt.text(
            j,
            i,
            text_val,
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if val > thresh else "black",
            fontsize=10,
        )

    plt.ylabel("True Class", fontsize=11, fontweight="semibold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="semibold")
    plt.tight_layout()

    try:
        plt.savefig(out, dpi=200)
        logger.debug("Saved confusion matrix plot to %s", out)
    except Exception as exc:
        logger.error("Failed to save confusion matrix plot to %s: %s", out, exc)
    finally:
        plt.close()
