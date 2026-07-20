"""
pathoai/fusion/summary.py
=========================
Spatial Fusion Summary Generator.

Compiles summary statistics of cell-ROI mapping distributions, spatial labels,
and distance distributions into markdown and text reports.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.9
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from pathoai.core.types import SpatialDetection


def generate_spatial_fusion_summary(
    spatial_detections: List[SpatialDetection]
) -> Dict[str, Any]:
    """Generates structured spatial fusion summary statistics.

    Parameters
    ----------
    spatial_detections : List[SpatialDetection]
        List of spatial detections.

    Returns
    -------
    Dict[str, Any]
        Summary report dictionary.
    """
    total = len(spatial_detections)
    if total == 0:
        return {
            "total_spatial_detections": 0,
            "label_distribution": {},
            "mean_boundary_distance_um": 0.0,
            "markdown_summary": "# Spatial Fusion Summary\n\nNo spatial detections.",
        }

    label_counts: Dict[str, int] = {}
    boundary_dists = []

    for sd in spatial_detections:
        lbl = sd.spatial_label
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        boundary_dists.append(sd.distance_to_tumor_boundary_um)

    mean_dist = float(np.mean(boundary_dists)) if boundary_dists else 0.0

    summary_text = (
        f"# Spatial Fusion Summary\n\n"
        f"- **Total Spatial Detections**: {total:,}\n"
        f"- **Mean Distance to Tumor Boundary**: {mean_dist:.2f} μm\n"
        f"- **Label Distribution**:\n"
    )
    for lbl, count in label_counts.items():
        summary_text += f"  - `{lbl}`: {count:,} ({count/total*100:.1f}%)\n"

    return {
        "total_spatial_detections": total,
        "label_distribution": label_counts,
        "mean_boundary_distance_um": mean_dist,
        "markdown_summary": summary_text,
    }
