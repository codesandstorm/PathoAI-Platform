"""
pathoai/detection/inference.py
==============================
Batch Inference Engine for Object Detection.

Executes detection inference over streaming batches of extracted image tiles.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.6
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from pathoai.detection.model import DetectionModel
from pathoai.detection.tiling import TileMetadata


class DetectionInference:
    """Executes batch inference across image tiles."""

    def __init__(
        self,
        model: DetectionModel,
        batch_size: int = 16,
        confidence_threshold: float = 0.25,
    ) -> None:
        """
        Parameters
        ----------
        model : DetectionModel
            Wrapped PyTorch detection model.
        batch_size : int
            Inference batch size.
        confidence_threshold : float
            Minimum score threshold to retain raw predictions.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive. Got: {batch_size}")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0, 1]. Got: {confidence_threshold}")

        self.model = model
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

    def predict_tiles(
        self, tiles: List[Tuple[TileMetadata, np.ndarray]]
    ) -> List[Tuple[TileMetadata, Dict[str, np.ndarray]]]:
        """Predict bounding boxes across a list of (TileMetadata, numpy_patch) pairs.

        Parameters
        ----------
        tiles : List[Tuple[TileMetadata, np.ndarray]]
            Tiles to process.

        Returns
        -------
        List[Tuple[TileMetadata, Dict[str, np.ndarray]]]
            List of (metadata, dict_of_boxes_scores_labels).
        """
        if not tiles:
            return []

        results = []
        n_tiles = len(tiles)

        for i in range(0, n_tiles, self.batch_size):
            batch_pairs = tiles[i : i + self.batch_size]
            batch_metas = [p[0] for p in batch_pairs]

            # Preprocess numpy patches (H, W, C) uint8 -> torch tensor (B, C, H, W) float32 [0, 1]
            batch_tensors = []
            for _, img in batch_pairs:
                t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                batch_tensors.append(t)

            batch_tensor = torch.stack(batch_tensors, dim=0)
            raw_outputs = self.model.predict_patch(batch_tensor)

            for meta, out in zip(batch_metas, raw_outputs):
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()

                # Apply confidence thresholding
                keep = scores >= self.confidence_threshold
                filtered_boxes = boxes[keep]
                filtered_scores = scores[keep]
                filtered_labels = labels[keep]

                results.append((
                    meta,
                    {
                        "boxes": filtered_boxes,
                        "scores": filtered_scores,
                        "labels": filtered_labels,
                    },
                ))

        return results
