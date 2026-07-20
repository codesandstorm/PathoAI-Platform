# Spatial Fusion Pipeline

The `FusionPipeline` ([pathoai/fusion/pipeline.py](file:///d:/Research/PathoAI-Platform/pathoai/fusion/pipeline.py)) coordinates the spatial fusion execution workflow.

---

## 🔄 Execution Steps

```
TumorROI + CellDetection
           │
           ▼
     ROIMapper (SpatialIndex candidate lookups & polygon checks)
           │
           ▼
     Distance Computation (Boundary & centroid distances)
           │
           ▼
     Spatial Classification (intratumoral / stromal / distant)
           │
           ▼
     SpatialValidator (Consistency verification)
           │
           ▼
     Typed SpatialDetection List Output
```
