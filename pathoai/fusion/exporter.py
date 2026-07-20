"""
pathoai/fusion/exporter.py
==========================
Spatial Fusion Exporter Engine.

Exports typed SpatialDetection domain objects into standard JSON and CSV format files.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.7
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Union

from pathoai.core.types import SpatialDetection


def export_spatial_detections_to_json(
    spatial_detections: List[SpatialDetection], output_path: Union[str, Path]
) -> None:
    """Exports list of SpatialDetection objects to JSON format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for sd in spatial_detections:
        records.append({
            "detection_id": sd.detection.detection_id,
            "slide_id": sd.detection.slide_id,
            "roi_id": str(sd.roi.roi_id),
            "class_id": sd.detection.class_id,
            "class_name": sd.detection.class_name,
            "confidence": float(sd.detection.confidence),
            "centroid_xy": sd.detection.centroid.to_tuple(),
            "inside_tumor": bool(sd.inside_tumor),
            "inside_stroma": bool(sd.inside_stroma),
            "distance_to_tumor_boundary_um": round(float(sd.distance_to_tumor_boundary_um), 3),
            "distance_to_roi_centroid_um": round(float(sd.distance_to_roi_centroid_um), 3),
            "nearest_boundary_point": sd.nearest_boundary_point.to_tuple(),
            "spatial_label": sd.spatial_label,
        })

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"spatial_detections": records}, f, indent=2)


def export_spatial_detections_to_csv(
    spatial_detections: List[SpatialDetection], output_path: Union[str, Path]
) -> None:
    """Exports list of SpatialDetection objects to CSV format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "detection_id", "slide_id", "roi_id", "class_id", "class_name",
        "confidence", "centroid_x", "centroid_y", "inside_tumor",
        "inside_stroma", "distance_to_tumor_boundary_um",
        "distance_to_roi_centroid_um", "nearest_boundary_x",
        "nearest_boundary_y", "spatial_label",
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sd in spatial_detections:
            writer.writerow({
                "detection_id": sd.detection.detection_id,
                "slide_id": sd.detection.slide_id,
                "roi_id": str(sd.roi.roi_id),
                "class_id": sd.detection.class_id,
                "class_name": sd.detection.class_name,
                "confidence": sd.detection.confidence,
                "centroid_x": sd.detection.centroid.x,
                "centroid_y": sd.detection.centroid.y,
                "inside_tumor": sd.inside_tumor,
                "inside_stroma": sd.inside_stroma,
                "distance_to_tumor_boundary_um": sd.distance_to_tumor_boundary_um,
                "distance_to_roi_centroid_um": sd.distance_to_roi_centroid_um,
                "nearest_boundary_x": sd.nearest_boundary_point.x,
                "nearest_boundary_y": sd.nearest_boundary_point.y,
                "spatial_label": sd.spatial_label,
            })
