"""
pathoai/detection/merger.py
===========================
Detection Tile Merger Engine.

Merges detections across overlapping patch tiles by mapping local box coordinates
to slide space and applying global Non-Maximum Suppression (NMS).

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.8
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from pathoai.detection.postprocessing import apply_nms
from pathoai.detection.tiling import TileMetadata


class DetectionMerger:
    """Merges overlapping detections from multiple tiles in slide-level space."""

    def __init__(self, iou_threshold: float = 0.45) -> None:
        """
        Parameters
        ----------
        iou_threshold : float
            Intersection over Union threshold for tile overlap merging.
        """
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be in [0, 1]. Got: {iou_threshold}")
        self.iou_threshold = iou_threshold

    def merge_tile_detections(
        self, tile_predictions: List[Tuple[TileMetadata, Dict[str, np.ndarray]]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maps local tile detections to slide space and merges duplicates using NMS.

        Parameters
        ----------
        tile_predictions : List[Tuple[TileMetadata, Dict[str, np.ndarray]]]
            List of (tile_metadata, dict_with_boxes_scores_labels).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Merged (slide_boxes, scores, labels) in level-0 pixel coordinates.
            slide_boxes shape: (N, 4) in format [x1, y1, x2, y2].
        """
        if not tile_predictions:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        all_slide_boxes = []
        all_scores = []
        all_labels = []

        for meta, pred in tile_predictions:
            boxes = pred.get("boxes", np.empty((0, 4)))
            scores = pred.get("scores", np.empty((0,)))
            labels = pred.get("labels", np.empty((0,)))

            if len(boxes) == 0:
                continue

            # Convert tile-local [x1, y1, x2, y2] to slide level-0 coordinates
            slide_boxes = boxes.copy()
            slide_boxes[:, 0] += meta.tile_x0
            slide_boxes[:, 1] += meta.tile_y0
            slide_boxes[:, 2] += meta.tile_x0
            slide_boxes[:, 3] += meta.tile_y0

            all_slide_boxes.append(slide_boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if not all_slide_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        cat_boxes = np.concatenate(all_slide_boxes, axis=0)
        cat_scores = np.concatenate(all_scores, axis=0)
        cat_labels = np.concatenate(all_labels, axis=0)

        # Execute global NMS on merged slide-level boxes
        merged_boxes, merged_scores, merged_labels = apply_nms(
            boxes=cat_boxes,
            scores=cat_scores,
            labels=cat_labels,
            iou_threshold=self.iou_threshold,
        )

        return merged_boxes, merged_scores, merged_labels
