"""
pathoai/fusion/coordinate_index.py
==================================
Spatial Bounding Box Index.

Provides fast spatial candidate lookup (grid index) for mapping cell centroid
coordinates to TumorROIs on large whole-slide images.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.3
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from pathoai.core.types import Point, TumorROI


class SpatialIndex:
    """Spatial grid index for fast candidate TumorROI lookups."""

    def __init__(self, grid_size: int = 1024) -> None:
        """
        Parameters
        ----------
        grid_size : int
            Size of spatial grid cells in level-0 pixels.
        """
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive. Got: {grid_size}")
        self.grid_size = grid_size
        self.grid: Dict[Tuple[int, int], List[TumorROI]] = {}
        self.rois: List[TumorROI] = []

    def build_index(self, rois: List[TumorROI]) -> None:
        """Populate spatial grid index with TumorROI bounding boxes.

        Parameters
        ----------
        rois : List[TumorROI]
            List of TumorROI objects.
        """
        self.rois = list(rois)
        self.grid.clear()

        for roi in self.rois:
            min_gx = roi.bbox.min_x // self.grid_size
            max_gx = roi.bbox.max_x // self.grid_size
            min_gy = roi.bbox.min_y // self.grid_size
            max_gy = roi.bbox.max_y // self.grid_size

            for gy in range(min_gy, max_gy + 1):
                for gx in range(min_gx, max_gx + 1):
                    key = (gx, gy)
                    if key not in self.grid:
                        self.grid[key] = []
                    self.grid[key].append(roi)

    def query_candidate_rois(self, point: Point) -> List[TumorROI]:
        """Find candidate TumorROIs whose bounding box overlaps the grid cell containing point.

        Parameters
        ----------
        point : Point
            Point coordinate at level 0.

        Returns
        -------
        List[TumorROI]
            Candidate TumorROI objects.
        """
        gx = int(point.x) // self.grid_size
        gy = int(point.y) // self.grid_size
        return self.grid.get((gx, gy), [])
