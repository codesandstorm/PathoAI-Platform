"""
tests/unit/tumor_bulk/test_exporters.py
=======================================
Unit tests for GeoJSON exporter.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import json
from pathlib import Path

from pathoai.tumor_bulk.exporters import export_rois_to_geojson


class TestExporters:
    """Test exporters."""

    def test_export_rois_to_geojson(self, tmp_path):
        """Test exporting ROIs list to a GeoJSON file."""
        rois = [
            {
                "roi_id": 1,
                "bbox_yxyx": [2, 2, 4, 4],
                "centroid_xy": (3.0, 3.0),
                "area_um2": 2.25,
                "perimeter_um": 12.0,
                "contours": [
                    [[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]
                ]
            }
        ]

        output_file = tmp_path / "test_annotation.geojson"
        export_rois_to_geojson(rois, output_file)

        assert output_file.is_file()

        # Read back and verify structure
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1

        feature = data["features"][0]
        assert feature["type"] == "Feature"
        assert feature["properties"]["roi_id"] == 1
        assert feature["properties"]["area_um2"] == 2.25
        assert feature["properties"]["perimeter_um"] == 12.0

        # Geometry checks
        geometry = feature["geometry"]
        assert geometry["type"] == "Polygon"
        # Ring should be closed automatically: first coordinate equals last
        expected_coordinates = [[[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0], [2.0, 2.0]]]
        assert geometry["coordinates"] == expected_coordinates
