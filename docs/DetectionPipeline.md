# Detection Pipeline

The `DetectionPipeline` ([pathoai/detection/pipeline.py](file:///d:/Research/PathoAI-Platform/pathoai/detection/pipeline.py)) coordinates cell detection processing.

---

## 🔄 Execution Workflow

1. **ROI Receiving**: Accepts `TumorROI` objects from Milestone 6.
2. **Tile Generation**: `TileGenerator` generates sliding-window patch tiles covering ROI bounds.
3. **Batch Inference**: `DetectionInference` runs batched predictions on target hardware (CPU/CUDA).
4. **Tile Overlap Merging**: `DetectionMerger` maps predictions to slide level-0 pixel coordinates and applies global IoU NMS deduplication.
5. **Coordinate Transformation**: `CoordinateTransformer` converts pixel metrics to physical $\mu m$ and instantiates typed `CellDetection` objects.
6. **Export & Visualization**: `exporter` and `visualization` modules format outputs into JSON, CSV, COCO, YOLO TXT, and overlay figures.
