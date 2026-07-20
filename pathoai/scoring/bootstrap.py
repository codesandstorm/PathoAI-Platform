"""
pathoai/scoring/bootstrap.py
============================
Bootstrap Confidence Interval Engine.

Performs bootstrap resampling over spatial detections to estimate empirical 95%
Confidence Intervals for sTIL scores.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.5
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from pathoai.core.types import FusionResult


class BootstrapEngine:
    """Estimates empirical confidence intervals using bootstrap resampling."""

    def __init__(self, n_iterations: int = 500, confidence_level: float = 0.95, seed: int = 42) -> None:
        """
        Parameters
        ----------
        n_iterations : int
            Number of bootstrap resampling iterations (e.g. 100, 500, 1000).
        confidence_level : float
            Target confidence level (default 0.95 for 95% CI).
        seed : int
            Random seed for reproducibility.
        """
        if n_iterations <= 0:
            raise ValueError(f"n_iterations must be positive. Got: {n_iterations}")
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.seed = seed

    def compute_confidence_interval(
        self, fusion_result: FusionResult, base_score: float
    ) -> Tuple[float, float]:
        """Computes bootstrap confidence interval bounds for sTIL score.

        Parameters
        ----------
        fusion_result : FusionResult
            Spatial fusion results container.
        base_score : float
            Primary sTIL score percentage.

        Returns
        -------
        Tuple[float, float]
            (ci_lower, ci_upper) bounds.
        """
        if not fusion_result.spatial_detections:
            return (base_score, base_score)

        rng = np.random.default_rng(self.seed)
        n_dets = len(fusion_result.spatial_detections)
        scores = []

        for _ in range(self.n_iterations):
            # Resample detections with replacement
            indices = rng.choice(n_dets, size=n_dets, replace=True)
            resampled = [fusion_result.spatial_detections[i] for i in indices]

            # Count resampled stromal cells
            resampled_stromal = sum(1 for sd in resampled if sd.inside_stroma)
            if n_dets > 0:
                score_inst = (resampled_stromal / n_dets) * (base_score * 2.0)
            else:
                score_inst = base_score
            scores.append(max(0.0, min(100.0, score_inst)))

        alpha = 1.0 - self.confidence_level
        lower_p = (alpha / 2.0) * 100.0
        upper_p = (1.0 - alpha / 2.0) * 100.0

        ci_lower = float(np.percentile(scores, lower_p))
        ci_upper = float(np.percentile(scores, upper_p))

        return (round(ci_lower, 2), round(ci_upper, 2))
