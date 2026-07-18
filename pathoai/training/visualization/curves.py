"""
pathoai/training/visualization/curves.py
=======================================
Matplotlib Training Curves Plotter.

Generates polished figures for train/val losses, learning rates, epoch durations,
and performance metrics over training runs.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.7
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import matplotlib.pyplot as plt
    # Trigger DLL loading early to catch AppLocker blocks
    import matplotlib.backends.backend_agg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import pandas as pd

from pathoai.core.logger import get_logger

logger = get_logger(__name__)



def plot_training_curves(
    history: Union[List[Dict[str, Any]], pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Plot training curves from training history data.

    Parameters
    ----------
    history : List[Dict[str, Any]] | pd.DataFrame
        List of epoch dictionaries or a Pandas DataFrame of training history.
    output_dir : str | Path
        Directory where curve images will be saved.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("plot_training_curves: Matplotlib is not available or blocked. Skipping curves plot.")
        return

    if isinstance(history, list):
        df = pd.DataFrame(history)
    else:
        df = history

    if df.empty:
        logger.warning("plot_training_curves: empty history dataframe. Skipping.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use a premium style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    epochs = df["epoch"]

    # 1. Loss Curve
    if "train_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2, marker="o")
        if "val_loss" in df.columns:
            plt.plot(epochs, df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, marker="s")
        plt.title("Training and Validation Loss", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out / "loss.png", dpi=200)
        plt.close()

    # Helper function to plot single metrics
    def _plot_metric(col_name: str, title: str, filename: str, color: str = "#2ca02c") -> None:
        if col_name in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, df[col_name], label=title, color=color, linewidth=2, marker="o")
            plt.title(title, fontsize=13, fontweight="bold", pad=12)
            plt.xlabel("Epoch", fontsize=11)
            plt.ylabel("Value", fontsize=11)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(out / filename, dpi=200)
            plt.close()

    # 2. Dice Curve
    _plot_metric("mean_dice", "Mean Dice Coefficient", "dice.png", "#2ca02c")

    # 3. IoU Curve
    _plot_metric("mean_iou", "Mean Intersection over Union (mIoU)", "iou.png", "#d62728")

    # 4. F1 Curve
    _plot_metric("macro_f1", "Macro F1 Score", "f1.png", "#9467bd")

    # 5. Precision & Recall Curves (Combined if both exist)
    if "weighted_precision" in df.columns and "weighted_recall" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["weighted_precision"], label="Precision", color="#8c564b", linewidth=2, marker="o")
        plt.plot(epochs, df["weighted_recall"], label="Recall", color="#e377c2", linewidth=2, marker="s")
        plt.title("Weighted Precision and Recall", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Score", fontsize=11)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out / "precision_recall.png", dpi=200)
        plt.close()
    else:
        _plot_metric("weighted_precision", "Weighted Precision", "precision.png", "#8c564b")
        _plot_metric("weighted_recall", "Weighted Recall", "recall.png", "#e377c2")

    # 6. Learning Rate Curve
    _plot_metric("learning_rate", "Learning Rate Schedule", "lr.png", "#bcbd22")

    # 7. Epoch Time Curve
    _plot_metric("elapsed_time", "Epoch Training Duration (seconds)", "epoch_time.png", "#17becf")

    logger.debug("Training curves saved to %s", out)
