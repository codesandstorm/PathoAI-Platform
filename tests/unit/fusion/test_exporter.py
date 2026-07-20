"""
tests/unit/fusion/test_exporter.py
===================================
Unit tests for spatial fusion exporters.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import json

from pathoai.core.types import BoundingBox, CellDetection, Point, SpatialDetection, TumorROI
from pathoai.fusion.exporter import (
    export_spatial_detections_to_csv,
    export_spatial_detections_to_json,
)


class TestSpatialExporter:
    """Test spatial fusion JSON and CSV exporters."""

    def test_exporters(self, tmp_path):
        """Test exporting SpatialDetection instances."""
        roi = TumorROI(1, BoundingBox(0, 0, 10, 10), Point(5.0, 5.0), 100, 25.0, 40.0, [])
        det = CellDetection("d1", "s1", "1", BoundingBox(0, 0, 5, 5), Point(2.5, 2.5), 0.9, 2, "lymphocyte")

        sd = SpatialDetection(
            detection=det,
            roi=roi,
            inside_tumor=True,
            inside_stroma=False,
            distance_to_tumor_boundary_um=0.0,
            distance_to_roi_centroid_um=3.5,
            nearest_boundary_point=Point(5.0, 0.0),
            spatial_label="intratumoral_lymphocyte",
        )

        out_json = tmp_path / "spatial.json"
        out_csv = tmp_path / "spatial.csv"

        export_spatial_detections_to_json([sd], out_json)
        export_spatial_detections_to_csv([sd], out_csv)

        assert out_json.is_file()
        assert out_csv.is_file()

        with open(out_json) as f:
            data = json.load(f)
            assert len(data["spatial_detections"]) == 1
