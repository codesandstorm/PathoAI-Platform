"""
pathoai/detection/pipeline.py
==============================
Cell Detection Pipeline Coordinator.

Orchestrates streaming patch tile generation, model batch inference, tile overlap
NMS merging, and coordinate transformation to return typed CellDetection domain objects.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.17
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from pathoai.core.types import CellDetection, TumorROI
from pathoai.detection.coordinate_transform import CoordinateTransformer
from pathoai.detection.factory import create_detector
from pathoai.detection.inference import DetectionInference
from pathoai.detection.merger import DetectionMerger
from pathoai.detection.model import DetectionModel
from pathoai.detection.tiling import TileGenerator


class DetectionPipeline:
    """High-level cell detection pipeline coordinator."""

    def __init__(
        self,
        model: Optional[DetectionModel] = None,
        config: Optional[Any] = None,
        tile_size: int = 640,
        overlap: int = 64,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        batch_size: int = 16,
    ) -> None:
        """
        Parameters
        ----------
        model : Optional[DetectionModel]
            Optional pre-instantiated detection model. If None, factory creates model from config.
        config : Optional[Any]
            ConfigNode configuration instance.
        tile_size : int
            Square tile width and height.
        overlap : int
            Overlap in pixels between adjacent tiles.
        confidence_threshold : float
            Minimum prediction confidence threshold.
        nms_iou_threshold : float
            Intersection over Union threshold for NMS merging.
        batch_size : int
            Batch size during inference.
        """
        if model is None:
            if config is None:
                # Default minimal config dict fallback
                config = {
                    "detection": {
                        "architecture": "yolo",
                        "n_classes": 4,
                        "in_channels": 3,
                    }
                }
            py_model = create_detector(config)
            self.model = DetectionModel(py_model)
        else:
            self.model = model

        self.tile_generator = TileGenerator(tile_size=tile_size, overlap=overlap)
        self.inference_engine = DetectionInference(
            model=self.model,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
        )
        self.merger = DetectionMerger(iou_threshold=nms_iou_threshold)

    def process_roi(
        self,
        roi: TumorROI,
        image: np.ndarray,
        slide_id: str,
        mpp: float,
    ) -> List[CellDetection]:
        """Runs the detection pipeline over a single TumorROI.

        Parameters
        ----------
        roi : TumorROI
            Target Region of Interest.
        image : np.ndarray
            RGB image array of shape (H, W, 3).
        slide_id : str
            Identifier of the source slide.
        mpp : float
            Microns per pixel resolution.

        Returns
        -------
        List[CellDetection]
            List of typed CellDetection objects.
        """
        # 1. Stream tiles from image corresponding to ROI bounds
        tiles = list(self.tile_generator.extract_tiles_from_array(image, roi))
        if not tiles:
            return []

        # 2. Execute batch inference over extracted tiles
        tile_predictions = self.inference_engine.predict_tiles(tiles)

        # 3. Merge tile-level detections across overlap borders using NMS
        merged_boxes, merged_scores, merged_labels = self.merger.merge_tile_detections(
            tile_predictions
        )

        if len(merged_boxes) == 0:
            return []

        # 4. Transform coordinates and construct typed CellDetection objects
        transformer = CoordinateTransformer(mpp=mpp)
        detections = transformer.create_cell_detections(
            slide_boxes=merged_boxes,
            scores=merged_scores,
            labels=merged_labels,
            slide_id=slide_id,
            roi_id=str(roi.roi_id),
        )

        return detections

    def process_rois(
        self,
        rois: List[TumorROI],
        image: np.ndarray,
        slide_id: str,
        mpp: float,
    ) -> List[CellDetection]:
        """Runs the detection pipeline across multiple TumorROIs.

        Parameters
        ----------
        rois : List[TumorROI]
            List of TumorROIs.
        image : np.ndarray
            RGB image array of shape (H, W, 3).
        slide_id : str
            Identifier of the source slide.
        mpp : float
            Microns per pixel resolution.

        Returns
        -------
        List[CellDetection]
            List of typed CellDetection objects across all ROIs.
        """
        all_detections = []
        for roi in rois:
            dets = self.process_roi(roi, image, slide_id, mpp)
            all_detections.extend(dets)
        return all_detections
