"""
tests/unit/detection/test_merger.py
===================================
Unit tests for DetectionMerger.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.detection.merger import DetectionMerger
from pathoai.detection.tiling import TileMetadata


class TestDetectionMerger:
    """Test DetectionMerger tile overlap deduplication."""

    def test_merge_tile_detections(self):
        """Test merging overlapping tile predictions into slide space."""
        meta1 = TileMetadata(0, "1", 0, 0, 64, 64, 0, 0)
        meta2 = TileMetadata(1, "1", 10, 0, 64, 64, 10, 0)

        pred1 = {
            "boxes": np.array([[5.0, 5.0, 15.0, 15.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([1]),
        }
        # Local tile box [0, 5, 10, 15] in tile2 maps to slide box [10, 5, 20, 15], overlaps with [5, 5, 15, 15]
        pred2 = {
            "boxes": np.array([[0.0, 5.0, 10.0, 15.0]]),
            "scores": np.array([0.85]),
            "labels": np.array([1]),
        }

        merger = DetectionMerger(iou_threshold=0.3)
        merged_boxes, merged_scores, merged_labels = merger.merge_tile_detections([
            (meta1, pred1),
            (meta2, pred2),
        ])

        assert len(merged_boxes) == 1
        assert merged_scores[0] == 0.9
