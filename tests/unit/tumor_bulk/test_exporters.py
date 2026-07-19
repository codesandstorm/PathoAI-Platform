"""
tests/unit/tumor_bulk/test_exporters.py
=======================================
Unit tests for GeoJSON exporter consuming TumorROI objects.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import json
from pathlib import Path

from pathoai.core.types import BoundingBox, Point, Polygon, TumorROI
from pathoai.tumor_bulk.exporters import export_rois_to_geojson


class TestExporters:
    """Test exporters."""

    def test_export_rois_to_geojson(self, tmp_path):
        """Test exporting ROIs list to a GeoJSON file."""
        rois = [
            TumorROI(
                roi_id=1,
                bbox=BoundingBox(min_y=2, min_x=2, max_y=4, max_x=4),
                centroid=Point(x=3.0, y=3.0),
                area_px=4,
                area_um2=2.25,
                perimeter_um=12.0,
                contours=[
                    Polygon(exterior=[
                        Point(2.0, 2.0),
                        Point(4.0, 2.0),
                        Point(4.0, 4.0),
                        Point(2.0, 4.0)
                    ])
                ],
                eccentricity=0.5,
                solidity=0.95,
                compactness=0.88,
                equivalent_diameter_um=1.5,
                class_label="tumor_bulk"
            )
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
        assert feature["properties"]["class_label"] == "tumor_bulk"
        assert feature["properties"]["area_um2"] == 2.25
        assert feature["properties"]["perimeter_um"] == 12.0
        assert feature["properties"]["eccentricity"] == 0.5
        assert feature["properties"]["solidity"] == 0.95
        assert feature["properties"]["compactness"] == 0.88
        assert feature["properties"]["equivalent_diameter_um"] == 1.5

        # Geometry checks
        geometry = feature["geometry"]
        assert geometry["type"] == "Polygon"
        # Ring should be closed automatically: first coordinate equals last
        expected_coordinates = [[[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0], [2.0, 2.0]]]
        assert geometry["coordinates"] == expected_coordinates
