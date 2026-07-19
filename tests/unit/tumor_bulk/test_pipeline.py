"""
tests/unit/tumor_bulk/test_pipeline.py
======================================
Unit tests for TumorBulkPipeline coordinator.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.tumor_bulk.pipeline import TumorBulkPipeline


class TestTumorBulkPipeline:
    """Test TumorBulkPipeline coordinate workflow."""

    def test_pipeline_validation(self):
        """Test validation on invalid constructor params."""
        with pytest.raises(ValueError, match="dilation_dist_um must be non-negative"):
            TumorBulkPipeline(dilation_dist_um=-1.0)
        with pytest.raises(ValueError, match="min_area_um2 must be non-negative"):
            TumorBulkPipeline(min_area_um2=-1.0)

    def test_pipeline_process_workflow(self):
        """Test process workflow on mock masks."""
        pipeline = TumorBulkPipeline(
            dilation_dist_um=1.0,  # 2 pixels radius
            min_area_um2=0.5,
            class_label="necrotic_region"
        )

        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[5, 5] = 1

        tumor_bed, rois = pipeline.process(mask, mpp=0.5)

        assert tumor_bed.shape == (15, 15)
        # Dilated bed should have 13 pixels (circle radius 2 around (5,5))
        assert np.sum(tumor_bed) == 13
        assert len(rois) == 1

        roi = rois[0]
        assert roi.roi_id == 1
        assert roi.class_label == "necrotic_region"
        assert roi.area_px == 13
        assert roi.area_um2 == 13 * 0.25
