"""
pathoai/validation/visualization.py
===================================
Validation Visualization Engine.

Generates publication-ready figures: Bland–Altman plots, scatter plots,
Precision-Recall curves, and score histograms.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.15
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ValidationVisualizer:
    """Generates publication-ready figures."""

    def plot_bland_altman(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Renders Bland–Altman agreement plot."""
        if not MATPLOTLIB_AVAILABLE:
            return

        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        means = (y_t + y_p) / 2.0
        diffs = y_p - y_t
        bias = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(means, diffs, color="royalblue", alpha=0.7, edgecolors="none")
        ax.axhline(bias, color="red", linestyle="--", label=f"Mean Bias: {bias:.2f}")
        ax.axhline(bias + 1.96 * std_diff, color="gray", linestyle=":", label="+1.96 SD")
        ax.axhline(bias - 1.96 * std_diff, color="gray", linestyle=":", label="-1.96 SD")

        ax.set_title("Bland–Altman Agreement Plot")
        ax.set_xlabel("Mean sTIL Score (%)")
        ax.set_ylabel("Difference (AI - Pathologist) (%)")
        ax.legend(loc="upper right")
        plt.tight_layout()

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300)
        plt.close(fig)

    def plot_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Renders AI vs Pathologist sTIL score scatter plot."""
        if not MATPLOTLIB_AVAILABLE:
            return

        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_t, y_p, color="crimson", alpha=0.7)
        ax.plot([0, 100], [0, 100], color="black", linestyle="--", label="Identity (1:1)")

        ax.set_title("AI vs Pathologist sTIL Scores")
        ax.set_xlabel("Pathologist sTIL Score (%)")
        ax.set_ylabel("AI sTIL Score (%)")
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.legend(loc="upper left")
        plt.tight_layout()

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300)
        plt.close(fig)
