"""
pathoai/fusion/aggregator.py
============================
Patch-to-Slide Level Aggregator.

Aggregates patch-level segmentations, cell counts, and scores into slide-level metrics
and constructs spatial grid heatmaps for sTIL density distributions.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.3
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class PatchAggregator:
    """Aggregates patch-level scores, coordinates, and metrics into slide-level results."""

    def __init__(self, stride: int) -> None:
        """
        Parameters
        ----------
        stride : int
            Patch extraction stride in level-0 pixels.
        """
        if stride <= 0:
            raise ValueError(f"stride must be positive. Got: {stride}")
        self.stride = stride
        self.patches: List[Dict[str, Any]] = []

    def add_patch(
        self,
        x_level0: int,
        y_level0: int,
        score: float,
        stroma_area_um2: float,
        n_lymphocytes: int,
    ) -> None:
        """Add a single patch result to the aggregator.

        Parameters
        ----------
        x_level0 : int
            x coordinate of patch top-left corner at level 0.
        y_level0 : int
            y coordinate of patch top-left corner at level 0.
        score : float
            Computed patch sTIL score (e.g. density or percentage).
        stroma_area_um2 : float
            Area of tumor-associated stroma inside the patch (in um^2).
        n_lymphocytes : int
            Number of stromal lymphocytes detected in the patch.
        """
        self.patches.append({
            "x": x_level0,
            "y": y_level0,
            "score": float(score),
            "stroma_area": float(stroma_area_um2),
            "n_lymphocytes": int(n_lymphocytes),
        })

    def aggregate(self) -> Dict[str, Any]:
        """Aggregate patch results into slide-level scores and construct the spatial heatmap.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing aggregated slide statistics:
            - "slide_score": weighted average sTIL score
            - "total_stroma_area_mm2": total stroma area across slide
            - "total_lymphocytes": total lymphocytes counted
            - "heatmap": 2D numpy array representing spatial grid of patch scores
        """
        if not self.patches:
            return {
                "slide_score": 0.0,
                "total_stroma_area_mm2": 0.0,
                "total_lymphocytes": 0,
                "heatmap": np.zeros((1, 1), dtype=np.float32),
            }

        # Calculate grid bounds
        xs = [p["x"] for p in self.patches]
        ys = [p["y"] for p in self.patches]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Convert coordinates to grid indices
        x_indices = [(x - min_x) // self.stride for x in xs]
        y_indices = [(y - min_y) // self.stride for y in ys]

        grid_w = max(x_indices) + 1
        grid_h = max(y_indices) + 1

        # Create heatmap grid
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        for p, xi, yi in zip(self.patches, x_indices, y_indices):
            heatmap[yi, xi] = p["score"]

        # Aggregate total stroma area and lymphocytes
        total_stroma_um2 = sum(p["stroma_area"] for p in self.patches)
        total_lymphocytes = sum(p["n_lymphocytes"] for p in self.patches)

        total_stroma_mm2 = total_stroma_um2 / 1_000_000.0

        # Calculate weighted average slide score
        # Patches with more stroma contribute more weight to the overall score
        total_weight = sum(p["stroma_area"] for p in self.patches)
        if total_weight > 0.0:
            weighted_sum = sum(p["score"] * p["stroma_area"] for p in self.patches)
            slide_score = weighted_sum / total_weight
        else:
            # Fallback to simple average if stroma area is zero everywhere
            slide_score = np.mean([p["score"] for p in self.patches])

        return {
            "slide_score": float(slide_score),
            "total_stroma_area_mm2": float(total_stroma_mm2),
            "total_lymphocytes": int(total_lymphocytes),
            "heatmap": heatmap,
        }
