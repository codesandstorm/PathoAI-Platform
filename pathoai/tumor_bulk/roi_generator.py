"""
pathoai/tumor_bulk/roi_generator.py
==================================
Tumor Bulk ROI Generator.

Extracts Region of Interest (ROI) metadata (bounding boxes, areas, centroids,
perimeters, eccentricity, solidity, compactness) into type-safe TumorROI models.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.4
"""

from __future__ import annotations

from typing import List

import numpy as np
import skimage.measure

from pathoai.core.types import BoundingBox, Point, Polygon, TumorROI
from pathoai.tumor_bulk.contours import extract_region_contours


def generate_rois(
    labeled_mask: np.ndarray,
    mpp: float,
    class_label: str = "tumor_bulk",
) -> List[TumorROI]:
    """Generates a list of TumorROI objects with advanced morphology metadata.

    Parameters
    ----------
    labeled_mask : np.ndarray
        Integer mask where 0 is background, and 1, 2, ... are separate regions.
    mpp : float
        Microns per pixel resolution.
    class_label : str
        Clinical classification label of the regions.

    Returns
    -------
    List[TumorROI]
        List of typed TumorROI objects.
    """
    if mpp <= 0:
        raise ValueError(f"mpp must be positive. Got: {mpp}")

    num_features = int(np.max(labeled_mask))
    if num_features == 0:
        return []

    # Compute regionprops
    props = skimage.measure.regionprops(labeled_mask)
    rois = []

    for prop in props:
        label_id = prop.label
        comp_mask = labeled_mask == label_id

        # Basic geometry
        min_y, min_x, max_y, max_x = prop.bbox
        bbox = BoundingBox(min_y=min_y, min_x=min_x, max_y=max_y, max_x=max_x)

        centroid_y, centroid_x = prop.centroid
        centroid = Point(x=centroid_x, y=centroid_y)

        area_px = int(prop.area)
        area_um2 = area_px * (mpp**2)

        # Perimeter in microns
        perimeter_px = float(prop.perimeter)
        perimeter_um = perimeter_px * mpp

        # Advanced region statistics
        eccentricity = float(prop.eccentricity)
        solidity = float(prop.solidity)
        # equivalent_diameter_area is preferred to avoid deprecation warnings
        if hasattr(prop, "equivalent_diameter_area"):
            eq_diam = prop.equivalent_diameter_area
        else:
            eq_diam = prop.equivalent_diameter
        equivalent_diameter_um = float(eq_diam) * mpp

        # Compactness: 4 * pi * Area / Perimeter^2
        if perimeter_px > 0.0:
            compactness = (4.0 * np.pi * area_px) / (perimeter_px**2)
        else:
            compactness = 0.0

        # Extract contours
        contours = extract_region_contours(comp_mask)
        polygon_contours = []
        for c in contours:
            pts = [Point(x=p[0], y=p[1]) for p in c]
            polygon_contours.append(Polygon(exterior=pts))

        roi = TumorROI(
            roi_id=label_id,
            bbox=bbox,
            centroid=centroid,
            area_px=area_px,
            area_um2=float(area_um2),
            perimeter_um=float(perimeter_um),
            contours=polygon_contours,
            eccentricity=eccentricity,
            solidity=solidity,
            compactness=compactness,
            equivalent_diameter_um=equivalent_diameter_um,
            class_label=class_label,
        )
        rois.append(roi)

    return rois
