# Geometry Engine

The Geometry Engine ([pathoai/fusion/geometry.py](file:///d:/Research/PathoAI-Platform/pathoai/fusion/geometry.py)) provides standard 2D computational geometry functions.

---

## 📐 Supported Operations

1. **`point_in_polygon(point, polygon)`**: Uses ray-casting algorithm to test whether a `Point` is inside a `Polygon` shell and outside interior holes.
2. **`nearest_boundary_point(point, polygon)`**: Projects a query point onto polygon exterior line segments to find the exact closest boundary coordinate.
3. **`distance_to_polygon(point, polygon)`**: Computes minimum Euclidean distance from a point to a polygon boundary in pixels.
4. **`polygon_area(polygon)`**: Evaluates polygon area using the Shoelace formula.
5. **`polygon_perimeter(polygon)`**: Computes exterior boundary perimeter in pixels.
