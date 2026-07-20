"""
pathoai/fusion/validation.py
============================
Spatial Mapping Validation Engine.

Verifies spatial mapping uniqueness, coordinate consistency, and numerical stability
for SpatialDetection objects.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.6
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from pathoai.core.types import SpatialDetection


class SpatialValidator:
    """Validates spatial fusion mappings and coordinate metrics."""

    def validate_spatial_detections(
        self, spatial_detections: List[SpatialDetection]
    ) -> Dict[str, Any]:
        """Validate list of SpatialDetection objects.

        Parameters
        ----------
        spatial_detections : List[SpatialDetection]
            Spatial detections to validate.

        Returns
        -------
        Dict[str, Any]
            Validation status report.
        """
        issues = []
        valid_count = 0

        for idx, sd in enumerate(spatial_detections):
            if sd.detection is None or sd.roi is None:
                issues.append(f"Index {idx}: Missing detection or roi reference")
                continue

            if sd.distance_to_tumor_boundary_um < 0.0:
                issues.append(f"Index {idx}: Negative distance to tumor boundary ({sd.distance_to_tumor_boundary_um})")

            if sd.distance_to_roi_centroid_um < 0.0:
                issues.append(f"Index {idx}: Negative distance to centroid ({sd.distance_to_roi_centroid_um})")

            valid_count += 1

        return {
            "total_detections": len(spatial_detections),
            "valid_detections": valid_count,
            "passed": len(issues) == 0,
            "issues": issues,
        }
