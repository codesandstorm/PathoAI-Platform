# ROI Mapper

The `ROIMapper` ([pathoai/fusion/roi_mapper.py](file:///d:/Research/PathoAI-Platform/pathoai/fusion/roi_mapper.py)) maps individual `CellDetection` objects to target `TumorROI` instances.

---

## 🔄 Mapping Workflow

1. **Candidate Retrieval**: `SpatialIndex` queries candidate ROIs whose bounding boxes overlap the cell's spatial grid cell.
2. **Point-in-Polygon Check**: Evaluates if the cell centroid falls inside tumor polygon contours.
3. **Distance Computation**: Computes Euclidean distance in microns ($\mu m$) to nearest tumor boundary points and ROI centroids.
4. **Spatial Label Assignment**: Classifies detections as `intratumoral_<class>`, `peritumoral_stromal_<class>`, or `distant_<class>`.
5. **SpatialDetection Packaging**: Instantiates typed `SpatialDetection` objects.
