"""
pathoai/fusion/roi_mapper.py
============================
Cell Detection to TumorROI Mapper Engine.

Assigns CellDetection objects to their parent TumorROI using spatial index prefiltering
and geometric point-in-polygon queries.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.5
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from pathoai.core.types import CellDetection, Point, SpatialDetection, TumorROI
from pathoai.fusion.coordinate_index import SpatialIndex
from pathoai.fusion.geometry import distance_to_polygon, nearest_boundary_point, point_in_polygon


class ROIMapper:
    """Maps CellDetection instances to target TumorROIs and calculates spatial metrics."""

    def __init__(self, mpp: float, max_distance_um: float = 1000.0, grid_size: int = 1024) -> None:
        """
        Parameters
        ----------
        mpp : float
            Microns per pixel resolution.
        max_distance_um : float
            Maximum distance threshold in microns for associating cells to an ROI.
        grid_size : int
            Grid size for spatial index.
        """
        if mpp <= 0:
            raise ValueError(f"mpp must be positive. Got: {mpp}")
        self.mpp = mpp
        self.max_distance_um = max_distance_um
        self.spatial_index = SpatialIndex(grid_size=grid_size)

    def map_detection_to_rois(
        self, detection: CellDetection, rois: List[TumorROI]
    ) -> Optional[SpatialDetection]:
        """Maps a single CellDetection object to the most relevant TumorROI.

        Parameters
        ----------
        detection : CellDetection
            The target cell detection.
        rois : List[TumorROI]
            Available TumorROI regions on the slide.

        Returns
        -------
        Optional[SpatialDetection]
            Populated SpatialDetection object, or None if no valid ROI association.
        """
        if not rois:
            return None

        # Build index if not built
        if len(self.spatial_index.rois) != len(rois):
            self.spatial_index.build_index(rois)

        pt = detection.centroid
        candidates = self.spatial_index.query_candidate_rois(pt)
        if not candidates:
            candidates = rois  # Fallback to full search if grid cell is empty

        best_roi: Optional[TumorROI] = None
        best_dist_um = float("inf")
        best_nearest_pt = pt
        inside_tumor = False
        inside_stroma = False

        for roi in candidates:
            # Check bounding box containment first
            if (roi.bbox.min_x <= pt.x <= roi.bbox.max_x) and (roi.bbox.min_y <= pt.y <= roi.bbox.max_y):
                # Detailed polygon check
                in_poly = False
                for poly in roi.contours:
                    if point_in_polygon(pt, poly):
                        in_poly = True
                        near_p = nearest_boundary_point(pt, poly)
                        dist_px = distance_to_polygon(pt, poly)
                        dist_um = dist_px * self.mpp
                        if dist_um < best_dist_um:
                            best_dist_um = dist_um
                            best_roi = roi
                            best_nearest_pt = near_p
                            inside_tumor = True
                        break

                if in_poly:
                    break

            # Distance to ROI centroid as secondary fallback
            dx = pt.x - roi.centroid.x
            dy = pt.y - roi.centroid.y
            dist_um = math.sqrt(dx * dx + dy * dy) * self.mpp

            if dist_um < best_dist_um:
                best_dist_um = dist_um
                best_roi = roi
                best_nearest_pt = roi.centroid

        if best_roi is None or best_dist_um > self.max_distance_um:
            return None

        # Distance to ROI centroid
        dist_centroid_um = math.sqrt((pt.x - best_roi.centroid.x)**2 + (pt.y - best_roi.centroid.y)**2) * self.mpp

        # Assign spatial label classification based on location
        if inside_tumor:
            spatial_label = f"intratumoral_{detection.class_name}"
        elif best_dist_um <= 100.0:
            inside_stroma = True
            spatial_label = f"peritumoral_stromal_{detection.class_name}"
        else:
            spatial_label = f"distant_{detection.class_name}"

        return SpatialDetection(
            detection=detection,
            roi=best_roi,
            inside_tumor=inside_tumor,
            inside_stroma=inside_stroma,
            distance_to_tumor_boundary_um=float(best_dist_um),
            distance_to_roi_centroid_um=float(dist_centroid_um),
            nearest_boundary_point=best_nearest_pt,
            spatial_label=spatial_label,
            metadata={"mpp": self.mpp},
        )

    def map_detections(
        self, detections: List[CellDetection], rois: List[TumorROI]
    ) -> List[SpatialDetection]:
        """Maps multiple CellDetection instances to TumorROIs."""
        self.spatial_index.build_index(rois)
        spatial_dets = []
        for det in detections:
            s_det = self.map_detection_to_rois(det, rois)
            if s_det is not None:
                spatial_dets.append(s_det)
        return spatial_dets
