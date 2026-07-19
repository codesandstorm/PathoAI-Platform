"""
pathoai/tumor_bulk/roi_generator.py
==================================
Tumor Bulk ROI Generator.

Extracts Region of Interest (ROI) metadata (bounding boxes, areas, centroids, perimeters)
for labeled connected components.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.4
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from pathoai.tumor_bulk.contours import extract_region_contours


def generate_rois(labeled_mask: np.ndarray, mpp: float) -> List[Dict[str, Any]]:
    """Generates a list of ROI dictionaries with metadata for each labeled component.

    Parameters
    ----------
    labeled_mask : np.ndarray
        Integer mask where 0 is background, and 1, 2, ... are separate regions.
    mpp : float
        Microns per pixel resolution.

    Returns
    -------
    List[Dict[str, Any]]
        List of ROI metadata dictionaries.
    """
    if mpp <= 0:
        raise ValueError(f"mpp must be positive. Got: {mpp}")

    num_features = int(np.max(labeled_mask))
    if num_features == 0:
        return []

    rois = []

    for label_id in range(1, num_features + 1):
        comp_mask = labeled_mask == label_id
        if not np.any(comp_mask):
            continue

        rows, cols = np.where(comp_mask)
        min_y, max_y = int(rows.min()), int(rows.max())
        min_x, max_x = int(cols.min()), int(cols.max())

        centroid_y = float(np.mean(rows))
        centroid_x = float(np.mean(cols))

        area_px = int(np.sum(comp_mask))
        area_um2 = area_px * (mpp**2)

        # Extract contours
        contours = extract_region_contours(comp_mask)

        # Calculate perimeter using contour lengths
        perimeter_px = 0.0
        for c in contours:
            if len(c) > 1:
                diff = np.diff(c, axis=0)
                # Euclidean distance sum
                perimeter_px += float(np.sum(np.sqrt(np.sum(diff**2, axis=1))))
                
                # Connect last point back to first
                last_to_first = c[0] - c[-1]
                perimeter_px += float(np.sqrt(np.sum(last_to_first**2)))

        perimeter_um = perimeter_px * mpp

        rois.append({
            "roi_id": label_id,
            "bbox_yxyx": [min_y, min_x, max_y, max_x],
            "centroid_xy": (centroid_x, centroid_y),
            "area_px": area_px,
            "area_um2": float(area_um2),
            "perimeter_um": float(perimeter_um),
            "contours": [c.tolist() for c in contours],
        })

    return rois
