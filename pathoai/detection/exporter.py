"""
pathoai/detection/exporter.py
=============================
Cell Detection Exporters.

Exports typed CellDetection domain objects into standard formats:
JSON, CSV, COCO Annotations, and YOLO TXT format files.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.10
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from pathoai.core.types import CellDetection


def export_to_json(detections: List[CellDetection], output_path: Union[str, Path]) -> None:
    """Exports list of CellDetection objects to JSON format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for d in detections:
        records.append({
            "detection_id": d.detection_id,
            "slide_id": d.slide_id,
            "roi_id": d.roi_id,
            "bbox_yxyx": d.bbox.to_yxyx(),
            "centroid_xy": d.centroid.to_tuple(),
            "confidence": float(d.confidence),
            "class_id": int(d.class_id),
            "class_name": d.class_name,
            "area_pixels": float(d.area_pixels),
            "area_um2": float(d.area_um2),
        })

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"detections": records}, f, indent=2)


def export_to_csv(detections: List[CellDetection], output_path: Union[str, Path]) -> None:
    """Exports list of CellDetection objects to CSV format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "detection_id", "slide_id", "roi_id", "class_id", "class_name",
        "confidence", "min_y", "min_x", "max_y", "max_x", "centroid_x",
        "centroid_y", "area_pixels", "area_um2",
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in detections:
            writer.writerow({
                "detection_id": d.detection_id,
                "slide_id": d.slide_id,
                "roi_id": d.roi_id,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "min_y": d.bbox.min_y,
                "min_x": d.bbox.min_x,
                "max_y": d.bbox.max_y,
                "max_x": d.bbox.max_x,
                "centroid_x": d.centroid.x,
                "centroid_y": d.centroid.y,
                "area_pixels": d.area_pixels,
                "area_um2": d.area_um2,
            })


def export_to_coco(
    detections: List[CellDetection],
    output_path: Union[str, Path],
    img_width: int = 1000,
    img_height: int = 1000,
) -> None:
    """Exports list of CellDetection objects to COCO dataset JSON format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    images = [{
        "id": 1,
        "width": img_width,
        "height": img_height,
        "file_name": "slide_level0.png",
    }]

    categories = []
    seen_classes = set()
    for d in detections:
        if d.class_id not in seen_classes:
            seen_classes.add(d.class_id)
            categories.append({
                "id": d.class_id,
                "name": d.class_name,
                "supercategory": "cell",
            })

    annotations = []
    for idx, d in enumerate(detections, start=1):
        x1 = d.bbox.min_x
        y1 = d.bbox.min_y
        w = d.bbox.width
        h = d.bbox.height
        annotations.append({
            "id": idx,
            "image_id": 1,
            "category_id": d.class_id,
            "bbox": [x1, y1, w, h],
            "area": d.area_pixels,
            "iscrowd": 0,
            "score": d.confidence,
        })

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)


def export_to_yolo_txt(
    detections: List[CellDetection],
    output_path: Union[str, Path],
    img_width: int = 1000,
    img_height: int = 1000,
) -> None:
    """Exports list of CellDetection objects to normalized YOLO TXT format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for d in detections:
        cx = (d.centroid.x) / img_width
        cy = (d.centroid.y) / img_height
        w = (d.bbox.width) / img_width
        h = (d.bbox.height) / img_height
        lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
