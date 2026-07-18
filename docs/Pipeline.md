# PathoAI-Platform: Pipeline Specification

> **Document Version**: 1.0.0
> **Date**: 2026-07-18
> **Status**: Milestone 1 — Infrastructure Design
> **References**: Salgado et al. (2015), TILs Working Group (2014), TIGER Dataset Documentation

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Stage 0: Pre-flight Validation](#2-stage-0-pre-flight-validation)
3. [Stage 1: WSI Ingestion](#3-stage-1-wsi-ingestion)
4. [Stage 2: Metadata Extraction](#4-stage-2-metadata-extraction)
5. [Stage 3: Thumbnail Generation](#5-stage-3-thumbnail-generation)
6. [Stage 4: Tissue Detection](#6-stage-4-tissue-detection)
7. [Stage 5: Patch Extraction](#7-stage-5-patch-extraction)
8. [Stage 6: Stain Normalization](#8-stage-6-stain-normalization)
9. [Stage 7: Semantic Tissue Segmentation](#9-stage-7-semantic-tissue-segmentation)
10. [Stage 8: Cell Detection](#10-stage-8-cell-detection)
11. [Stage 9: Spatial Fusion and sTIL Computation](#11-stage-9-spatial-fusion-and-stil-computation)
12. [Stage 10: Statistical Validation](#12-stage-10-statistical-validation)
13. [Stage 11: Visualization](#13-stage-11-visualization)
14. [Stage 12: Report Generation](#14-stage-12-report-generation)
15. [Error Recovery and Resumption](#15-error-recovery-and-resumption)
16. [Pipeline Configuration Reference](#16-pipeline-configuration-reference)

---

## 1. Pipeline Overview

The PathoAI pipeline is a **strictly sequential, checkpoint-aware** processing pipeline. Each stage:

1. **Validates its inputs** before processing
2. **Produces named outputs** stored to disk (enabling resumption)
3. **Logs a completion event** with timing and quality metrics
4. **Writes a stage manifest** (JSON) recording all parameters used

```
Input: WSI File (.svs / .ndpi / .tif / .scn / .mrxs)
  │
  ▼ Stage 0: Pre-flight Validation
  │   ─ File exists, format supported, size within limits
  │   ─ GPU/CPU availability confirmed
  │   ─ Model checkpoints verified (if not training)
  │
  ▼ Stage 1: WSI Ingestion
  │   ─ OpenSlide reader open
  │   ─ File integrity verified (SHA-256)
  │
  ▼ Stage 2: Metadata Extraction
  │   ─ WSIMetadata dataclass populated
  │   ─ MPP validated (must be in acceptable range)
  │   ─ Magnification confirmed
  │
  ▼ Stage 3: Thumbnail Generation
  │   ─ Low-resolution overview image (level 2 or 3)
  │   ─ Saved as PNG for quality inspection
  │
  ▼ Stage 4: Tissue Detection
  │   ─ Convert thumbnail to HSV
  │   ─ Apply Otsu thresholding on saturation channel
  │   ─ Morphological cleanup (opening, closing)
  │   ─ Tissue mask (binary) at thumbnail resolution
  │   ─ Quality gate: tissue ratio must exceed minimum threshold
  │
  ▼ Stage 5: Patch Extraction
  │   ─ Sliding window at target magnification level
  │   ─ Only tissue-positive patches extracted
  │   ─ PatchCoordinateMap written to disk
  │   ─ PatchDataset (PyTorch) instantiated in memory
  │
  ▼ Stage 6: Stain Normalization
  │   ─ Macenko/Vahadane normalization applied per patch
  │   ─ Reference stain vector from TIGER training set
  │
  ▼ Stage 7: Semantic Tissue Segmentation  [MODEL REQUIRED]
  │   ─ DeepLabV3+ inference on each patch
  │   ─ 6-class probability maps generated
  │   ─ Argmax → class maps
  │   ─ Reassembled into slide-level segmentation mask
  │
  ▼ Stage 8: Cell Detection               [MODEL REQUIRED]
  │   ─ Faster R-CNN inference on each patch
  │   ─ Bounding boxes, class labels, scores
  │   ─ NMS applied per patch
  │   ─ Coordinates mapped back to slide coordinate system
  │
  ▼ Stage 9: Spatial Fusion + sTIL Score
  │   ─ Cell centroids intersected with stroma mask
  │   ─ N_lymphocytes_in_stroma counted
  │   ─ Stroma area computed in μm²
  │   ─ sTIL score computed per sTIL equation
  │   ─ Patch-level scores aggregated to slide score
  │
  ▼ Stage 10: Statistical Validation
  │   ─ Bootstrap confidence intervals
  │   ─ Quality flags set if CI is too wide
  │   ─ Pathologist agreement computed (if labels available)
  │
  ▼ Stage 11: Visualization
  │   ─ Segmentation overlay image
  │   ─ Detection overlay image
  │   ─ sTIL spatial heatmap
  │   ─ Summary publication figure
  │
  ▼ Stage 12: Report Generation
      ─ JSON result with full provenance
      ─ PDF clinical summary card (future)
      ─ Update batch CSV
```

---

## 2. Stage 0: Pre-flight Validation

### Purpose
Fail fast before any expensive computation.

### Checks Performed

| Check | Condition | Action on Failure |
|-------|-----------|------------------|
| File existence | `os.path.exists(slide_path)` | Raise `WSIReadError` |
| File extension | Extension in supported list | Raise `WSIReadError` |
| File size | `> MIN_SIZE_BYTES (1 MB)` | Raise `DataError` |
| File size | `< MAX_SIZE_BYTES (50 GB)` | Raise `DataError` with warning |
| Compute device | `torch.cuda.is_available()` | Log warning, fallback to CPU |
| Segmentation checkpoint | File exists + SHA matches | Raise `CheckpointLoadError` if inference mode |
| Detection checkpoint | File exists + SHA matches | Raise `CheckpointLoadError` if inference mode |
| Output directory | Writable | Raise `IOError` |
| Disk space | `> MIN_FREE_GB (5 GB)` | Raise `EnvironmentError` |

### Output
- `preflight_manifest.json` written to output directory
- Stage marked PASSED or FAILED

---

## 3. Stage 1: WSI Ingestion

### Purpose
Open the WSI file using OpenSlide and prepare the reader object.

### Processing Steps

1. Attempt `openslide.OpenSlide(slide_path)`
2. On `openslide.OpenSlideError`: retry once, then raise `WSIReadError`
3. Compute SHA-256 hash of file (streamed, 64 KB chunks to avoid RAM spike)
4. Store reader object in WSI engine context

### Technical Notes
- OpenSlide `open()` is lazy — it reads headers only, not pixel data
- Reader is kept open for the duration of the slide's pipeline run
- Reader is explicitly closed in `finally` block to prevent handle leaks

### Output
- OpenSlide reader object (in memory)
- `file_sha256: str`

---

## 4. Stage 2: Metadata Extraction

### Purpose
Extract all scientifically relevant metadata from WSI headers.

### Properties Extracted

```python
# Via openslide.OpenSlide properties dict:

slide_id          # filename stem
format            # openslide.PROPERTY_NAME_VENDOR
width, height     # level_dimensions[0]
mpp_x             # openslide.PROPERTY_NAME_MPP_X  (μm/pixel)
mpp_y             # openslide.PROPERTY_NAME_MPP_Y
objective_power   # openslide.PROPERTY_NAME_OBJECTIVE_POWER
n_levels          # level_count
level_dimensions  # list of (W, H) per level
level_downsamples # list of float downsample factors
vendor            # scanner manufacturer
```

### MPP Validation

```
Acceptable MPP range at 40x: 0.20 – 0.30 μm/pixel
Acceptable MPP range at 20x: 0.40 – 0.60 μm/pixel
```

If MPP is outside acceptable range OR missing:
- Log `WARNING` with actual value
- Attempt heuristic estimation from `objective_power` property
- If still unavailable, raise `DataError` (MPP is required for area computation)

### Output
- `WSIMetadata` dataclass (frozen, immutable)
- Serialized to `metadata.json` in output directory

---

## 5. Stage 3: Thumbnail Generation

### Purpose
Generate a low-resolution overview image for tissue detection and quality inspection.

### Algorithm

```python
# Select the thumbnail level:
# Prefer level where longest dimension < THUMBNAIL_MAX_DIM (default: 2048px)
thumbnail_level = wsi_engine.select_thumbnail_level(slide, max_dim=2048)
thumbnail_array = np.array(slide.read_region((0, 0), thumbnail_level,
                                              slide.level_dimensions[thumbnail_level]))
# Convert RGBA → RGB (OpenSlide returns RGBA)
thumbnail_rgb = thumbnail_array[:, :, :3]
```

### Output
- `thumbnail.png` — RGB PNG saved to output directory
- `thumbnail_level: int` — level used
- `thumbnail_mpp: float` — MPP at thumbnail level

---

## 6. Stage 4: Tissue Detection

### Purpose
Generate a binary mask separating tissue from background (glass, pen marks, artifacts).

### Algorithm: Otsu Thresholding in HSV Space

**Rationale**: H&E stained tissue appears pink/purple with high color saturation. Background glass appears white with very low saturation. Thresholding the Saturation channel in HSV space robustly separates these.

```python
# 1. Convert thumbnail RGB to HSV
hsv = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)
saturation = hsv[:, :, 1]

# 2. Apply Otsu's method to saturation channel
threshold, tissue_binary = cv2.threshold(
    saturation, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 3. Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
tissue_mask = cv2.morphologyEx(tissue_binary, cv2.MORPH_CLOSE, kernel)
tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

# 4. Remove small connected components (artifact blobs)
tissue_mask = remove_small_objects(tissue_mask.astype(bool),
                                    min_size=MIN_TISSUE_COMPONENT_PIXELS)
```

### Quality Gate

```python
tissue_ratio = tissue_mask.sum() / tissue_mask.size
if tissue_ratio < cfg.wsi.tissue_detection.min_tissue_ratio:
    raise DataError(f"Insufficient tissue: {tissue_ratio:.3f} < {min_ratio}")
```

Default `min_tissue_ratio` = 0.05 (5% of slide area must be tissue).

### Output
- `tissue_mask.npy` — boolean array at thumbnail resolution
- `tissue_ratio: float`
- `otsu_threshold: int` — logged for reproducibility

---

## 7. Stage 5: Patch Extraction

### Purpose
Extract square image patches from tissue regions at the target magnification level.

### Target Magnification Selection

```python
# Find pyramid level closest to target magnification
target_mpp = SEGMENTATION_TARGET_MPP  # e.g., 0.50 for 20x equivalent
extraction_level = wsi_engine.find_best_level(slide, target_mpp)
```

### Sliding Window Algorithm

```python
# Upsample tissue mask to level 0 coordinates for precise filtering
for y in range(0, slide_height - patch_size, stride):
    for x in range(0, slide_width - patch_size, stride):
        # Check if patch center falls in tissue mask
        thumb_x = int(x / downsample_x)
        thumb_y = int(y / downsample_y)
        if tissue_mask[thumb_y, thumb_x]:
            coordinates.append((x, y))
```

### Patch Reading

```python
region = slide.read_region(
    location=(x, y),            # Coordinates at level 0
    level=extraction_level,
    size=(patch_size, patch_size)
)
patch_rgb = np.array(region)[:, :, :3]  # Drop alpha channel
```

### Quality Filters per Patch

| Filter | Condition | Action |
|--------|-----------|--------|
| Blank patch | Mean pixel intensity > 230 (pure white) | Skip |
| Blurry patch | Laplacian variance < BLUR_THRESHOLD | Skip with warning |
| Artifact | Ink/pen detection heuristic | Skip with warning |

### Output
- `PatchCoordinateMap` — serialized to `patch_coords.json`
- `PatchDataset` — PyTorch Dataset object
- Statistics: `n_total_patches`, `n_tissue_patches`, `n_skipped_patches`

---

## 8. Stage 6: Stain Normalization

### Purpose
Normalize H&E staining variation across scanners and batches to improve model generalization.

### Default Method: Macenko (2009)

**Rationale**: Macenko method is computationally efficient and widely used in computational pathology. It decomposes the staining matrix via SVD and normalizes against a reference stain vector derived from the training set.

```python
# Reference stain vector derived from TIGER training subset
REFERENCE_STAIN_MATRIX = np.array([
    [0.5626, 0.7201, 0.4062],  # Hematoxylin
    [0.2159, 0.8012, 0.5581],  # Eosin
])
```

### Fallback
If normalization fails (degenerate tissue, extreme staining): log WARNING and use unnormalized patch. Never silently corrupt the patch.

### Output
- Normalized patch tensors replacing raw patches in `PatchDataset`
- `normalization_method: str` logged in manifest

---

## 9. Stage 7: Semantic Tissue Segmentation

**Requires trained DeepLabV3+ checkpoint — implemented in Milestone 4**

### Input
- `PatchDataset` (normalized patches, batch_size configurable)
- Segmentation model checkpoint

### Processing

```python
# Per-batch inference:
with torch.no_grad():
    logits = model(patch_batch)         # (B, n_classes, H, W)
    probs = torch.softmax(logits, dim=1)
    pred_classes = probs.argmax(dim=1)  # (B, H, W)
```

### Slide-Level Mask Assembly
Patch predictions are stitched back into the slide coordinate system using `PatchCoordinateMap`, producing:
- `stroma_mask`: binary mask, 1 where class==STROMA_CLASS_ID
- `tumor_mask`: binary mask, 1 where class==TUMOR_CLASS_ID

### Output
- `segmentation_map.npy` — class predictions at patch resolution mapped to slide grid
- `stroma_mask.npy` — binary
- `tumor_mask.npy` — binary
- Per-patch class probability saved if `save_patch_probs=True`

---

## 10. Stage 8: Cell Detection

**Requires trained Faster R-CNN checkpoint — implemented in Milestone 5**

### Input
- `PatchDataset` re-sampled at 40× equivalent (higher resolution for cell-level detail)
- Detection model checkpoint

### Processing

```python
# Per-batch inference:
with torch.no_grad():
    predictions = model(patch_batch)
    # predictions: List[Dict] with 'boxes', 'labels', 'scores'

# Apply NMS (already integrated in Faster R-CNN head)
# Filter by minimum confidence score
filtered = [p for p in predictions if p['scores'] > cfg.detection.min_confidence]
```

### Coordinate Normalization
All bounding box coordinates are converted from patch-local pixels to slide-level coordinates using the `PatchCoordinateMap`.

### Output
- `List[CellDetectionResult]` — one per patch
- `detections_summary.json` — count by class

---

## 11. Stage 9: Spatial Fusion and sTIL Computation

### sTIL Scoring Methodology

Following Salgado et al. (2015) and the TILs Working Group Recommendations:

> "sTILs are defined as the percentage of the stromal area that is occupied by mononuclear inflammatory cells (including lymphocytes and plasma cells)."

**Implementation** (automated approximation using object counts):

```
sTIL_score = (N_lymphocytes_within_stroma / stroma_area_mm²) × normalization_factor
```

Where:
- `N_lymphocytes_within_stroma`: lymphocytes whose bounding box centroid falls on a stroma pixel
- `stroma_area_mm²`: computed from stroma mask pixel count × (MPP × 1e-3)²
- `normalization_factor`: calibrated against pathologist labels on TIGER validation set

### Patch-Level Processing

```python
for patch in tissue_patches:
    # Get stroma mask for this patch (from segmentation)
    stroma_mask_patch = get_patch_mask(stroma_mask, patch.coords, patch.size)

    # Get detections for this patch
    detections = cell_detections[patch.id]

    # Count lymphocytes in stroma
    lymphocyte_boxes = detections.boxes[detections.labels == LYMPHOCYTE_CLASS_ID]
    centroids = (lymphocyte_boxes[:, :2] + lymphocyte_boxes[:, 2:]) / 2
    in_stroma = [stroma_mask_patch[int(cy), int(cx)] for cx, cy in centroids]
    n_lymph_stroma = sum(in_stroma)

    # Compute stroma area
    stroma_area_px = stroma_mask_patch.sum()
    stroma_area_um2 = stroma_area_px * (mpp ** 2)

    patch.stil_score = n_lymph_stroma / (stroma_area_um2 / 1e6)  # per mm²
```

### Slide-Level Aggregation

```python
# Weighted mean by stroma area (patches with more stroma contribute more)
weights = np.array([p.stroma_area_um2 for p in tissue_patches])
scores = np.array([p.stil_score for p in tissue_patches])
slide_stil = np.average(scores, weights=weights)
```

### Output
- `sTILResult` dataclass
- `stil_heatmap.npy` — per-patch scores mapped to slide grid

---

## 12. Stage 10: Statistical Validation

### Bootstrap Confidence Interval

```python
bootstrap_scores = []
for _ in range(cfg.validation.bootstrap_n):
    sample = np.random.choice(patch_scores, size=len(patch_scores), replace=True)
    bootstrap_scores.append(np.average(sample, weights=sample_weights))

ci_lower = np.percentile(bootstrap_scores, 2.5)
ci_upper = np.percentile(bootstrap_scores, 97.5)
```

### Quality Flags

| Flag | Condition | Meaning |
|------|-----------|---------|
| `LOW_CONFIDENCE` | CI width > 20 percentage points | Score uncertain; flag for pathologist review |
| `INSUFFICIENT_STROMA` | Stroma area < MIN_STROMA_AREA_MM2 | Not enough stroma to score reliably |
| `INSUFFICIENT_LYMPHOCYTES` | N < 50 lymphocytes detected | Detection may be unreliable |
| `SCORE_BOUNDARY` | sTIL near clinical cutoffs (10%, 20%) | Borderline case; flag for review |

### Output
- `ValidationReport` with statistics, CI, and flags
- `validation_summary.json`

---

## 13. Stage 11: Visualization

### Outputs Generated

| File | Description |
|------|-------------|
| `thumbnail_tissue.png` | Thumbnail + tissue mask overlay (green) |
| `segmentation_overlay.png` | H&E patch + color-coded class overlay |
| `detection_overlay.png` | H&E patch + bounding boxes per class |
| `stil_heatmap.png` | Slide-level spatial sTIL density heatmap |
| `summary_figure.png` | 4-panel publication figure |

### Class Color Map (standardized):

| Class | Color (RGB) | Hex |
|-------|-------------|-----|
| background | (220, 220, 220) | #DCDCDC |
| tumor_bulk | (255, 80, 80) | #FF5050 |
| stroma | (80, 200, 120) | #50C878 |
| lymphocytes | (80, 120, 255) | #5078FF |
| necrosis | (200, 160, 80) | #C8A050 |
| other | (180, 180, 180) | #B4B4B4 |

---

## 14. Stage 12: Report Generation

### JSON Report Schema

```json
{
  "schema_version": "1.0.0",
  "slide_id": "TIGER-001-Slide",
  "processed_at": "2026-07-18T17:09:45+05:30",
  "pipeline_version": "1.0.0",
  "config_hash": "sha256:abcdef...",
  "still_score_pct": 34.7,
  "confidence_interval_95": [28.2, 41.3],
  "quality_flags": [],
  "stroma_area_mm2": 12.45,
  "n_lymphocytes_in_stroma": 4320,
  "n_tissue_patches": 847,
  "seg_model": {"name": "deeplabv3plus_efficientnet_b3", "checkpoint_sha": "sha256:..."},
  "det_model": {"name": "faster_rcnn_resnet34", "checkpoint_sha": "sha256:..."},
  "wsi_metadata": { ... },
  "processing_time_breakdown": {
    "stage_1_ingestion_s": 0.3,
    "stage_2_metadata_s": 0.1,
    "stage_3_thumbnail_s": 0.5,
    "stage_4_tissue_s": 0.8,
    "stage_5_patches_s": 28.4,
    "stage_6_normalization_s": 45.2,
    "stage_7_segmentation_s": 180.5,
    "stage_8_detection_s": 120.3,
    "stage_9_fusion_s": 5.1,
    "stage_10_validation_s": 2.2,
    "stage_11_visualization_s": 8.7,
    "total_s": 392.1
  }
}
```

---

## 15. Error Recovery and Resumption

### Checkpoint Mechanism

At each stage completion, a `stage_manifest.json` is written:

```json
{
  "stage": 5,
  "stage_name": "patch_extraction",
  "status": "COMPLETED",
  "completed_at": "2026-07-18T17:09:45+05:30",
  "output_files": ["patch_coords.json"],
  "parameters_hash": "sha256:..."
}
```

### Resumption Logic

```python
# On pipeline start, check for existing stage manifests
last_completed_stage = orchestrator.check_resume_point(output_dir)
if last_completed_stage > 0:
    logger.info(f"Resuming from Stage {last_completed_stage + 1}")
    orchestrator.load_stage_outputs(output_dir, up_to_stage=last_completed_stage)
```

Slides that fail a stage are logged to `failed_slides.csv` with the error message and stage number, enabling batch retry.

---

## 16. Pipeline Configuration Reference

```yaml
# config/base.yaml — Complete pipeline configuration defaults

pipeline:
  version: "1.0.0"
  seed: 42
  device: "auto"          # "auto" | "cuda" | "cpu"
  mixed_precision: false  # Set true only on CUDA with Tensor Cores
  resume: true            # Resume from checkpoint if output exists
  save_intermediates: true  # Save per-stage outputs (debugging)

wsi:
  supported_formats: [".svs", ".ndpi", ".tif", ".tiff", ".scn", ".mrxs"]
  thumbnail_max_dim: 2048

  tissue_detection:
    method: "otsu_hsv"
    min_tissue_ratio: 0.05
    morph_kernel_size: 15
    min_component_pixels: 1000

  patch_extraction:
    patch_size: 512
    stride: 256
    target_mpp: 0.50           # 20x equivalent
    blank_threshold: 230       # Mean RGB > this → blank
    blur_threshold: 50.0       # Laplacian variance < this → blurry

stain_normalization:
  method: "macenko"            # "macenko" | "vahadane" | "reinhard" | "none"
  reference_slide: null        # Path to reference slide for stain matrix

segmentation:
  model_name: "deeplabv3plus_efficientnet_b3"
  n_classes: 6
  input_size: 512
  batch_size: 8
  checkpoint: null             # Path to .pth checkpoint

detection:
  model_name: "faster_rcnn_resnet34"
  n_classes: 4                 # background + 3 cell classes
  input_size: 256
  batch_size: 16
  checkpoint: null
  min_confidence: 0.50
  nms_iou_threshold: 0.50
  target_mpp: 0.25             # 40x equivalent

fusion:
  lymphocyte_class_id: 2
  stroma_class_id: 1
  min_stroma_area_mm2: 0.5

validation:
  bootstrap_n: 1000
  confidence_level: 0.95
  min_lymphocytes_for_confidence: 50

logging:
  level: "INFO"
  format: "structured"
  file_rotation_mb: 50
  console: true

output:
  base_dir: "results/"
  save_segmentation_overlay: true
  save_detection_overlay: true
  save_heatmap: true
  save_json_report: true
  save_patch_probs: false      # Large files — disable unless debugging
```

---

*End of Pipeline Specification v1.0.0*
*Next: See Dataset_Specification.md for TIGER dataset integration details*
