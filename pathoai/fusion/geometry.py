"""
pathoai/fusion/geometry.py
==========================
Geometry Engine for Spatial Reasoning.

Provides robust 2D geometry functions: point-in-polygon queries, nearest boundary
points, point-to-polygon distances, polygon perimeters, areas, and mask calculations.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.4
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np

from pathoai.core.types import Point, Polygon


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting algorithm to check if a point lies inside a polygon shell.

    Parameters
    ----------
    point : Point
        Query coordinate.
    polygon : Polygon
        Target polygon.

    Returns
    -------
    bool
        True if point lies inside exterior shell and outside interior holes.
    """
    x, y = point.x, point.y
    shell = polygon.exterior
    n = len(shell)
    if n < 3:
        return False

    inside = False
    p1x, p1y = shell[0].x, shell[0].y
    for i in range(n + 1):
        p2x, p2y = shell[i % n].x, shell[i % n].y
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    if not inside:
        return False

    # Check holes
    for hole in polygon.interiors:
        if _point_in_ring(point, hole):
            return False

    return True


def _point_in_ring(point: Point, ring: List[Point]) -> bool:
    """Helper ray-casting for closed ring."""
    x, y = point.x, point.y
    n = len(ring)
    if n < 3:
        return False
    inside = False
    p1x, p1y = ring[0].x, ring[0].y
    for i in range(n + 1):
        p2x, p2y = ring[i % n].x, ring[i % n].y
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def nearest_boundary_point(point: Point, polygon: Polygon) -> Point:
    """Finds the closest point on the polygon's exterior boundary segments to the query point.

    Parameters
    ----------
    point : Point
        Query coordinate.
    polygon : Polygon
        Target polygon.

    Returns
    -------
    Point
        Nearest point on the polygon boundary.
    """
    shell = polygon.exterior
    if not shell:
        return point

    if len(shell) < 2:
        return shell[0]

    min_dist_sq = float("inf")
    closest_pt = shell[0]

    px, py = point.x, point.y

    for i in range(len(shell)):
        p1 = shell[i]
        p2 = shell[(i + 1) % len(shell)]

        # Segment projection math
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            nx, ny = x1, y1
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            nx = x1 + t * dx
            ny = y1 + t * dy

        dist_sq = (px - nx) ** 2 + (py - ny) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_pt = Point(x=nx, y=ny)

    return closest_pt


def distance_to_polygon(point: Point, polygon: Polygon) -> float:
    """Computes minimum Euclidean distance from a point to a polygon boundary in pixels.

    Parameters
    ----------
    point : Point
        Query coordinate.
    polygon : Polygon
        Target polygon.

    Returns
    -------
    float
        Distance in pixel units.
    """
    near_pt = nearest_boundary_point(point, polygon)
    return math.sqrt((point.x - near_pt.x) ** 2 + (point.y - near_pt.y) ** 2)


def polygon_area(polygon: Polygon) -> float:
    """Computes area of polygon using Shoelace formula in pixels^2."""
    shell = polygon.exterior
    n = len(shell)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += shell[i].x * shell[j].y
        area -= shell[j].x * shell[i].y
    area = abs(area) / 2.0

    for hole in polygon.interiors:
        if len(hole) >= 3:
            h_area = 0.0
            hn = len(hole)
            for i in range(hn):
                j = (i + 1) % hn
                h_area += hole[i].x * hole[j].y
                h_area -= hole[j].x * hole[i].y
            area -= abs(h_area) / 2.0

    return max(0.0, float(area))


def polygon_perimeter(polygon: Polygon) -> float:
    """Computes perimeter of polygon exterior boundary in pixels."""
    shell = polygon.exterior
    n = len(shell)
    if n < 2:
        return 0.0
    perim = 0.0
    for i in range(n):
        p1 = shell[i]
        p2 = shell[(i + 1) % n]
        perim += math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    return float(perim)


def calculate_mask_area(mask: np.ndarray, mpp: float) -> float:
    """Computes physical area of a binary mask in square millimeters (mm^2)."""
    if mpp <= 0:
        raise ValueError(f"mpp must be positive. Got: {mpp}")

    pixel_count = int(np.sum(mask > 0))
    area_mm2 = (pixel_count * (mpp**2)) / 1_000_000.0
    return float(area_mm2)
