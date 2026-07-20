# Detection Architecture (Milestone 7)

The Cell Detection Engine (`pathoai/detection/`) implements a production-grade, modular, registry-driven framework for identifying individual cells (e.g. lymphocytes, cancer cells) inside whole-slide tissue regions.

---

## 🏗️ Core Architecture Components

```
TumorROI
   │
   ▼
DetectionPipeline
   ├── TileGenerator (Streaming patches over ROI bounds)
   ├── Detector (Registry-driven PyTorch model, e.g. YOLO)
   ├── DetectionInference (Batch execution)
   ├── PostProcessor (Confidence filter)
   ├── DetectionMerger (IoU NMS tile overlap deduplication)
   ├── CoordinateTransformer (Tile → ROI → Slide → μm)
   ├── Exporter (JSON, CSV, COCO, YOLO TXT)
   └── Visualizer (Overlays & density heatmaps)
```

---

## 🔑 Key Features
1. **Decoupled Architecture Registry**: Uses `@register_detector` to register model architectures. Adding future models (e.g. RT-DETR, CellViT, HoVer-Net) requires zero changes to downstream code.
2. **Typed Domain Objects**: Always returns typed `CellDetection` dataclasses containing `bbox`, `centroid`, `confidence`, `class_name`, and physical `area_um2`.
3. **Memory-Conscious Streaming**: `TileGenerator` streams patches over ROI bounding boxes without loading full WSIs into RAM.
4. **Tile Overlap Merging**: `DetectionMerger` handles deduplication across adjacent tile margins using IoU Non-Maximum Suppression (NMS).
