"""
tests/unit/fusion/test_geometry.py
===================================
Unit tests for Geometry Engine.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import pytest

from pathoai.core.types import Point, Polygon
from pathoai.fusion.geometry import (
    distance_to_polygon,
    nearest_boundary_point,
    point_in_polygon,
    polygon_area,
    polygon_perimeter,
)


class TestGeometryEngine:
    """Test geometry functions."""

    def test_point_in_polygon(self):
        """Test ray casting point in polygon."""
        poly = Polygon(exterior=[
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ])

        assert point_in_polygon(Point(5.0, 5.0), poly) is True
        assert point_in_polygon(Point(15.0, 5.0), poly) is False

    def test_nearest_boundary_point(self):
        """Test finding nearest boundary point."""
        poly = Polygon(exterior=[
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ])

        near_p = nearest_boundary_point(Point(12.0, 5.0), poly)
        assert near_p.x == 10.0
        assert near_p.y == 5.0

    def test_distance_to_polygon(self):
        """Test calculating distance to polygon."""
        poly = Polygon(exterior=[
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ])

        dist = distance_to_polygon(Point(15.0, 5.0), poly)
        assert dist == 5.0

    def test_polygon_area_and_perimeter(self):
        """Test area and perimeter calculations."""
        poly = Polygon(exterior=[
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ])

        assert polygon_area(poly) == 100.0
        assert polygon_perimeter(poly) == 40.0
