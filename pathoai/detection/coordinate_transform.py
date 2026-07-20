"""
pathoai/detection/coordinate_transform.py
==========================================
Coordinate Transformation System for Cell Detection.

Provides coordinate system transformations across frames:
Tile-Local Pixels → ROI Pixels → WSI Level-0 Pixels → Physical Microns (μm).
Produces typed CellDetection domain objects.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.9
"""

from __future__ import annotations

import uuid
from typing import List, Tuple

import numpy as np

from pathoai.core.constants import CELL_CLASSES
from pathoai.core.types import BoundingBox, CellDetection, Point, TumorROI


class CoordinateTransformer:
    """Handles coordinate frame transformations and typed CellDetection object creation."""

    def __init__(self, mpp: float) -> None:
        """
        Parameters
        ----------
        mpp : float
            Microns per pixel resolution at slide level-0.
        """
        if mpp <= 0:
            raise ValueError(f"mpp must be positive. Got: {mpp}")
        self.mpp = mpp

    def tile_to_slide_box(
        self, tile_box: np.ndarray, tile_x0: int, tile_y0: int
    ) -> np.ndarray:
        """Transform tile-local box [x1, y1, x2, y2] to slide level-0 coordinates."""
        slide_box = tile_box.copy()
        slide_box[0] += tile_x0
        slide_box[1] += tile_y0
        slide_box[2] += tile_x0
        slide_box[3] += tile_y0
        return slide_box

    def compute_centroid(self, box: np.ndarray) -> Point:
        """Calculate Point centroid (cx, cy) from box [x1, y1, x2, y2]."""
        cx = float((box[0] + box[2]) / 2.0)
        cy = float((box[1] + box[3]) / 2.0)
        return Point(x=cx, y=cy)

    def compute_physical_area_um2(self, box: np.ndarray) -> float:
        """Calculate physical box area in square microns (μm^2)."""
        w_px = max(0.0, float(box[2] - box[0]))
        h_px = max(0.0, float(box[3] - box[1]))
        return (w_px * self.mpp) * (h_px * self.mpp)

    def create_cell_detections(
        self,
        slide_boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        slide_id: str,
        roi_id: str,
    ) -> List[CellDetection]:
        """Convert slide-level prediction arrays into typed CellDetection objects.

        Parameters
        ----------
        slide_boxes : np.ndarray
            Boxes array of shape (N, 4) in level-0 pixel coordinates [x1, y1, x2, y2].
        scores : np.ndarray
            Scores array of shape (N,).
        labels : np.ndarray
            Class integer labels of shape (N,).
        slide_id : str
            Identifier of the source slide.
        roi_id : str
            Identifier of the parent ROI region.

        Returns
        -------
        List[CellDetection]
            List of typed CellDetection domain objects.
        """
        detections = []

        for i in range(len(slide_boxes)):
            box = slide_boxes[i]
            score = float(scores[i])
            label_id = int(labels[i])
            class_name = CELL_CLASSES.get(label_id, f"cell_class_{label_id}")

            min_x, min_y, max_x, max_y = int(np.round(box[0])), int(np.round(box[1])), int(np.round(box[2])), int(np.round(box[3]))
            bbox = BoundingBox(min_y=min_y, min_x=min_x, max_y=max_y, max_x=max_x)
            centroid = self.compute_centroid(box)

            w_px = max(0.0, float(max_x - min_x))
            h_px = max(0.0, float(max_y - min_y))
            area_px = w_px * h_px
            area_um2 = self.compute_physical_area_um2(box)

            det_id = f"det_{slide_id}_{roi_id}_{i:05d}_{uuid.uuid4().hex[:6]}"

            detections.append(
                CellDetection(
                    detection_id=det_id,
                    slide_id=slide_id,
                    roi_id=str(roi_id),
                    bbox=bbox,
                    centroid=centroid,
                    confidence=score,
                    class_id=label_id,
                    class_name=class_name,
                    area_pixels=area_px,
                    area_um2=area_um2,
                )
            )

        return detections
