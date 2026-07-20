"""
tests/unit/detection/test_inference.py
=======================================
Unit tests for DetectionInference engine.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np
import pytest

from pathoai.detection.factory import create_detector
from pathoai.detection.inference import DetectionInference
from pathoai.detection.model import DetectionModel
from pathoai.detection.tiling import TileMetadata


class TestDetectionInference:
    """Test detection inference engine."""

    def test_predict_tiles(self):
        """Test prediction over list of patch tiles."""
        config = {"detection": {"architecture": "yolo", "n_classes": 4}}
        py_model = create_detector(config)
        model = DetectionModel(py_model)

        engine = DetectionInference(model=model, batch_size=2, confidence_threshold=0.0)

        meta1 = TileMetadata(0, "1", 0, 0, 64, 64, 0, 0)
        meta2 = TileMetadata(1, "1", 64, 0, 64, 64, 64, 0)
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        tiles = [(meta1, img), (meta2, img)]
        results = engine.predict_tiles(tiles)

        assert len(results) == 2
        m, pred = results[0]
        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred
