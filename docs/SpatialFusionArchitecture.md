# Spatial Fusion Architecture (Milestone 8)

The Spatial Fusion Engine (`pathoai/fusion/`) connects extracted tissue regions (`TumorROI` objects from Milestone 6) with detected cells (`CellDetection` objects from Milestone 7) to determine spatial relationships.

---

## 🏗️ Core Architecture Components

```
TumorROI + CellDetection
           │
           ▼
     FusionPipeline
           ├── SpatialIndex (Grid-based candidate ROI lookups)
           ├── ROIMapper (Assigns detections to TumorROIs)
           ├── GeometryEngine (Point-in-polygon, boundary distances)
           ├── SpatialValidator (Consistency and numerical checks)
           ├── SpatialExporter (JSON and CSV formats)
           └── Visualizer (ROI boundaries & distance overlays)
```

---

## 🔑 Key Features

1. **Typed Domain Model**: Generates `SpatialDetection` instances linking `CellDetection` and `TumorROI` with spatial properties (`inside_tumor`, `inside_stroma`, `distance_to_tumor_boundary_um`, `distance_to_roi_centroid_um`, `nearest_boundary_point`, `spatial_label`).
2. **Spatial Indexing**: `SpatialIndex` provides $O(1)$ grid candidate lookups for handling large WSIs with thousands of detections.
3. **Geometry Engine**: Robust point-in-polygon, boundary projection, and distance calculation routines.
4. **Strict Scope Isolation**: Focuses purely on spatial geometry. sTIL percentage scoring is handled separately by Milestone 9.
