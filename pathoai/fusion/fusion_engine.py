"""
pathoai/fusion/fusion_engine.py
===============================
Spatial Fusion and sTIL scoring Engine.

Integrates semantic segmentation masks and cell detection coordinates to extract
the tumor bed, compute tumor-associated stroma areas, filter lymphocytes,
perform bootstrap confidence interval checks, and generate clinical flags.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.4
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from pathoai.core.exceptions import ValidationError
from pathoai.fusion.spatial_ops import (
    calculate_mask_area,
    extract_tumor_associated_stroma,
    extract_tumor_bed,
    filter_points_in_mask,
)
from pathoai.fusion.stil_computer import compute_stil_score


class FusionEngine:
    """Orchestrates spatial intersection logic and sTIL score derivation."""

    def __init__(
        self,
        mpp: float,
        dilation_dist_um: float = 500.0,
        bootstrap_n: int = 1000,
        min_stroma_area_mm2: float = 0.5,
        min_lymph_for_confidence: int = 50,
        lymphocyte_diameter_um: float = 10.0,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        mpp : float
            Microns per pixel resolution of the segmentation masks.
        dilation_dist_um : float
            Margin around tumor nests to define the tumor bed.
        bootstrap_n : int
            Number of iterations for confidence interval bootstrap resampling.
        min_stroma_area_mm2 : float
            Minimum stroma area required to score without a flag.
        min_lymph_for_confidence : int
            Minimum lymphocytes required to score without a flag.
        lymphocyte_diameter_um : float
            Lymphocyte diameter in microns for area calculations.
        seed : int
            Random seed for bootstrap reproducibility.
        """
        if mpp <= 0:
            raise ValidationError(f"mpp must be positive. Got: {mpp}")
        self.mpp = mpp
        self.dilation_dist_um = dilation_dist_um
        self.bootstrap_n = bootstrap_n
        self.min_stroma_area_mm2 = min_stroma_area_mm2
        self.min_lymph_for_confidence = min_lymph_for_confidence
        self.lymphocyte_diameter_um = lymphocyte_diameter_um
        self.seed = seed

    def process_slide(
        self,
        tumor_mask: np.ndarray,
        stroma_mask: np.ndarray,
        lymphocyte_centroids: np.ndarray | List[Tuple[float, float]],
        patch_coords: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Runs the complete tumor bulk extraction and sTIL computation pipeline.

        Parameters
        ----------
        tumor_mask : np.ndarray
            Binary segmentation mask where 1 indicates invasive tumor cells.
        stroma_mask : np.ndarray
            Binary segmentation mask where 1 indicates raw stroma cells.
        lymphocyte_centroids : np.ndarray | List[Tuple[float, float]]
            List or array of lymphocyte coordinates (x, y) at level 0.
        patch_coords : List[Dict[str, Any]] | None
            Optional list of patch dictionaries containing {"x": ..., "y": ..., "score": ...}
            used to perform patch-level bootstrap resampling.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing derived slide-level sTIL metrics.
        """
        if tumor_mask.shape != stroma_mask.shape:
            raise ValidationError(
                f"Shape mismatch: tumor_mask {tumor_mask.shape} vs stroma_mask {stroma_mask.shape}"
            )

        # 1. Extract contiguous Tumor Bed
        tumor_bed = extract_tumor_bed(
            tumor_mask=tumor_mask,
            mpp=self.mpp,
            dilation_dist_um=self.dilation_dist_um,
        )

        # 2. Extract Tumor-Associated Stroma
        tumor_associated_stroma = extract_tumor_associated_stroma(
            tumor_bed_mask=tumor_bed,
            stroma_mask=stroma_mask,
        )

        # 3. Calculate Stroma Area
        stroma_area_mm2 = calculate_mask_area(
            mask=tumor_associated_stroma,
            mpp=self.mpp,
        )

        # 4. Filter Lymphocytes to retain only those in tumor-associated stroma
        n_lymphocytes_total = len(lymphocyte_centroids)
        # downsample = self.mpp / level0_mpp
        level0_mpp = 0.25
        downsample = self.mpp / level0_mpp
        filtered_centroids, n_lymph_in_stroma = filter_points_in_mask(
            points=lymphocyte_centroids,
            mask=tumor_associated_stroma,
            downsample=downsample,
        )

        # 5. Compute sTIL Score Statistics
        score_stats = compute_stil_score(
            n_lymphocytes=n_lymph_in_stroma,
            stroma_area_mm2=stroma_area_mm2,
            lymphocyte_diameter_um=self.lymphocyte_diameter_um,
        )

        # 6. Bootstrap Confidence Intervals
        ci_lower, ci_upper = self._calculate_bootstrap_ci(
            patch_coords=patch_coords,
            fallback_score=score_stats["estimated_pct"],
        )

        # 7. Assign Quality Flags
        quality_flags = self._assign_quality_flags(
            stroma_area_mm2=stroma_area_mm2,
            n_lymph_in_stroma=n_lymph_in_stroma,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            estimated_pct=score_stats["estimated_pct"],
        )

        return {
            "tumor_bed_mask": tumor_bed,
            "tumor_associated_stroma_mask": tumor_associated_stroma,
            "stroma_area_mm2": stroma_area_mm2,
            "n_lymphocytes_total": n_lymphocytes_total,
            "n_lymphocytes_in_stroma": n_lymph_in_stroma,
            "filtered_centroids": filtered_centroids,
            "density_per_mm2": score_stats["density_per_mm2"],
            "sTIL_score_pct": score_stats["estimated_pct"],
            "ci_95": (ci_lower, ci_upper),
            "quality_flags": quality_flags,
        }

    def _calculate_bootstrap_ci(
        self,
        patch_coords: List[Dict[str, Any]] | None,
        fallback_score: float,
    ) -> Tuple[float, float]:
        """Computes the 95% confidence interval using bootstrap resampling of patches."""
        if not patch_coords or len(patch_coords) < 3:
            # Fall back to returning the point estimate if bootstrap data is insufficient
            return fallback_score, fallback_score

        rng = np.random.default_rng(self.seed)
        scores = np.array([p["score"] for p in patch_coords])
        weights = np.array([p.get("stroma_area", 1.0) for p in patch_coords])

        bootstrap_means = []
        n_patches = len(scores)

        for _ in range(self.bootstrap_n):
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

    def _assign_quality_flags(
        self,
        stroma_area_mm2: float,
        n_lymph_in_stroma: int,
        ci_lower: float,
        ci_upper: float,
        estimated_pct: float,
    ) -> List[str]:
        """Assigns clinical quality flags based on guidelines."""
        flags = []

        if stroma_area_mm2 < self.min_stroma_area_mm2:
            flags.append("INSUFFICIENT_STROMA")

        if n_lymph_in_stroma < self.min_lymph_for_confidence:
            flags.append("INSUFFICIENT_LYMPHOCYTES")

        ci_width = ci_upper - ci_lower
        if ci_width > 20.0:
            flags.append("LOW_CONFIDENCE")

        # Clinical boundary cases (near 10% or 20% cutoffs)
        for boundary in [10.0, 20.0]:
            if abs(estimated_pct - boundary) <= 2.0:
                flags.append("SCORE_BOUNDARY")
                break

        return flags
