"""
pathoai/stil/bootstrap.py
=========================
Bootstrap Resampling Engine.

Performs patch-level bootstrap resampling to compute 95% Confidence Intervals (CI).

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 9.3
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def calculate_bootstrap_ci(
    patch_coords: List[Dict[str, Any]] | None,
    fallback_score: float,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes the 95% confidence interval using bootstrap resampling of patches.

    Parameters
    ----------
    patch_coords : List[Dict[str, Any]] | None
        Optional list of patch dictionaries containing {"score": ..., "stroma_area": ...}.
    fallback_score : float
        Value to return if bootstrap data is insufficient.
    bootstrap_n : int
        Number of bootstrap resampling iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        95% Confidence Interval (ci_lower, ci_upper).
    """
    if not patch_coords or len(patch_coords) < 3:
        return fallback_score, fallback_score

    rng = np.random.default_rng(seed)
    scores = np.array([p["score"] for p in patch_coords])
    weights = np.array([p.get("stroma_area", 1.0) for p in patch_coords])

    bootstrap_means = []
    n_patches = len(scores)

    for _ in range(bootstrap_n):
        indices = rng.choice(n_patches, size=n_patches, replace=True)
        sampled_scores = scores[indices]
        sampled_weights = weights[indices]

        total_w = np.sum(sampled_weights)
        if total_w > 0.0:
            mean = np.sum(sampled_scores * sampled_weights) / total_w
        else:
            mean = np.mean(sampled_scores)
        bootstrap_means.append(mean)

    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    return ci_lower, ci_upper
