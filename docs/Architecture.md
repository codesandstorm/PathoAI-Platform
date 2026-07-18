# PathoAI-Platform: System Architecture

> **Document Version**: 1.0.0
> **Date**: 2026-07-18
> **Status**: Milestone 1 — Infrastructure Design
> **Classification**: Research-Grade Computational Pathology Pipeline

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Engine Descriptions](#4-engine-descriptions)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [Module Dependency Graph](#6-module-dependency-graph)
7. [Inter-Engine Communication Protocol](#7-inter-engine-communication-protocol)
8. [Configuration Architecture](#8-configuration-architecture)
9. [Logging and Observability Architecture](#9-logging-and-observability-architecture)
10. [Error Handling Strategy](#10-error-handling-strategy)
11. [Testing Architecture](#11-testing-architecture)
12. [Extensibility Design](#12-extensibility-design)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Security and Data Privacy](#14-security-and-data-privacy)
15. [Performance Design Targets](#15-performance-design-targets)
16. [Known Limitations and Future Work](#16-known-limitations-and-future-work)

---

## 1. Executive Summary

PathoAI-Platform is a modular, research-grade, end-to-end artificial intelligence pipeline for automated stromal Tumor Infiltrating Lymphocyte (sTIL) scoring from Whole Slide Images (WSIs) of breast cancer tissue.

The platform is designed as a publication-quality scientific software system intended for:

- **Research reproducibility**: All experiments, configurations, and results are fully logged and versioned.
- **Clinical translation readiness**: Clean interfaces, validated outputs, and auditable decision pathways.
- **Modular extensibility**: Each engine operates independently and can be replaced, upgraded, or parallelized without breaking adjacent components.
- **Community adoption**: Open-source design with comprehensive documentation enabling external researchers to reproduce results without contacting the authors.

The platform implements the methodology from published breast cancer sTIL literature (Salgado et al., 2015; TILs Working Group, 2014; Klauschen et al., 2018), replacing proprietary datasets with the publicly available TIGER (Tumor InfiltratinG lymphocytes in breast cancER) dataset from the Grand Challenge.

**Scientific Reference Pipeline**:
- Salgado R. et al. (2015). The evaluation of tumor-infiltrating lymphocytes (TILs) in breast cancer. *Annals of Oncology*, 26(2), 259–271.
- Klauschen F. et al. (2018). Scoring of tumor-infiltrating lymphocytes: From visual estimation to machine learning. *Seminars in Cancer Biology*, 52, 151–157.
- TIGER Grand Challenge: https://tiger.grand-challenge.org/

---

## 2. Design Philosophy

### Principle 1: Separation of Concerns
Each engine (WSI, Segmentation, Detection, Fusion, Validation, Visualization, Report) encapsulates a single responsibility. No engine has direct knowledge of another engine's internals. They communicate exclusively through well-defined data contracts (Python dataclasses).

### Principle 2: Configuration as Code
Every tunable parameter — from patch size to learning rate to staining normalization coefficients — lives in YAML configuration files versioned alongside the code. No magic numbers in source files. Every constant has a name and lives in `constants.py`.

### Principle 3: Reproducibility by Default
- All random seeds are controllable via configuration.
- All data loading operations are deterministic given the same seed.
- Environment snapshots (package versions, hardware specs) are generated at experiment start.
- Results are serialized with provenance metadata (timestamps, model checksums, config hashes).

### Principle 4: Fail Fast, Fail Clearly
Validation gates exist at every engine boundary. Invalid inputs are rejected with descriptive, actionable error messages before expensive computation begins. Silent degradation is strictly forbidden.

### Principle 5: Test Everything
Every public function has a corresponding unit test. Integration tests cover cross-engine data flows. Synthetic data generators eliminate dependency on real WSIs for CI/CD pipelines.

### Principle 6: Document Why, Not Just What
All architecture decisions are documented with rationale. Inline comments explain domain-specific pathology concepts for software engineers, and engineering concepts for pathologists and clinicians reviewing the code.

### Principle 7: Clinical Pathology Awareness
The software is built with full understanding of the clinical context:
- sTIL scoring follows standardized guidelines (TILs Working Group)
- Tissue compartment definitions align with pathologist consensus
- Score outputs are interpretable and traceable to source evidence

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PathoAI-Platform                                │
│                       Platform Core (Core Layer)                         │
│   Config │ Logger │ Constants │ Exceptions │ Utils │ Validators          │
│           Registry │ Types │ PipelineOrchestrator                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │         WSI Engine           │
                │  OpenSlide Multi-format Reader│
                │  Metadata Extraction          │
                │  Thumbnail Generation         │
                │  Tissue Detection (Otsu/HSV)  │
                │  Tissue-aware Patch Extraction│
                │  Coordinate Mapping           │
                └──────────────┬───────────────┘
                               │
                         [PatchDataset]
                         [PatchCoordinateMap]
                               │
                ┌──────────────▼───────────────┐
                │      Segmentation Engine      │
                │  Stain Normalization           │
                │  DeepLabV3+ (EfficientNet-B3) │
                │  6-class Tissue Classification│
                │  Slide-level Mask Assembly    │
                └────────┬──────────────────────┘
                         │
              [StromaMask]  [TumorMask]
                         │
                ┌────────▼──────────────────────┐
                │      Detection Engine          │
                │  Faster R-CNN (ResNet34 + FPN) │
                │  Cancer Cell Detection         │
                │  Lymphocyte Detection          │
                │  NMS + Coordinate Normalization│
                └──────────────┬────────────────┘
                               │
                        [CellDetectionResult]
                               │
                ┌──────────────▼────────────────┐
                │     Spatial Fusion Engine      │
                │  Mask × Detection Overlay      │
                │  sTIL Equation Solver          │
                │  Patch-level Aggregation       │
                │  Slide-level Score Computation │
                └──────────────┬────────────────┘
                               │
                         [sTILResult]
                               │
                ┌──────────────▼────────────────┐
                │      Validation Engine         │
                │  Bootstrap Confidence Intervals│
                │  Inter-rater Agreement (κ)    │
                │  Distribution Analysis         │
                │  Pass/Fail Gates               │
                └──────────────┬────────────────┘
                               │
                       [ValidatedResult]
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼──────────┐ ┌────────▼──────────┐ ┌───────▼──────────┐
│ Visualization     │ │  Report Engine    │ │ Clinical Dashboard│
│ Engine            │ │                   │ │ (Future Milestone)│
│ Heatmaps          │ │ JSON Export       │ │ Web UI / REST API │
│ Segmentation OL   │ │ PDF Clinical Card │ │ DICOM Integration │
│ Detection OL      │ │ CSV Batch Results │ │ HL7 FHIR Output   │
│ Publication Figs  │ │ Provenance Report │ └──────────────────┘
└───────────────────┘ └───────────────────┘
```

---

## 4. Engine Descriptions

### 4.1 Platform Core
**Location**: `pathoai/core/`
**Responsibility**: Cross-cutting concerns shared by all engines. No business logic lives here.

| Module | File | Responsibility |
|--------|------|---------------|
| Config Manager | `config.py` | YAML loading, env-var override, schema validation, merging base + experiment configs |
| Structured Logger | `logger.py` | Multi-handler logging with file rotation, structured JSON format, experiment context injection |
| Global Constants | `constants.py` | All named constants (MPP targets, class IDs, color maps, file extensions) |
| Exception Hierarchy | `exceptions.py` | Custom exception classes for typed error handling |
| Type Definitions | `types.py` | TypedDict, Protocol, and dataclass definitions shared across engines |
| General Utilities | `utils/` | IO, math, image ops, hashing, timing decorators |
| Input/Output Validators | `validators.py` | Pydantic-style validation for engine I/O contracts |
| Model Registry | `registry.py` | Decorator-based registry for dynamic model class resolution |
| Pipeline Orchestrator | `pipeline.py` | Wires engines together; manages experiment lifecycle |

---

### 4.2 WSI Engine
**Location**: `pathoai/wsi/`
**Responsibility**: Reading, parsing, and preprocessing Whole Slide Images into structured patch datasets.

**Input**: WSI file path (any OpenSlide-supported format)
**Output**: `PatchDataset` + `WSIMetadata` + `PatchCoordinateMap`

**Key Design Decisions**:

| Decision | Rationale |
|----------|-----------|
| OpenSlide as reader | Supports all major scanner formats; battle-tested in computational pathology community |
| Tissue detection at low resolution | Level 2-3 thumbnails are sufficient for morphology; avoids loading full gigapixel image |
| Otsu thresholding in HSV space | More robust than RGB for H&E stained tissue; separates background white from tissue purple-pink |
| Stride-based patch extraction | Configurable overlap prevents boundary artifacts; stride < patch_size gives overlap |
| Coordinate tracking in PatchCoordinateMap | Enables result reconstruction to slide coordinates without reprocessing |
| No ML in WSI engine | Keeps this engine fast, testable without GPU, and maintainable |

---

### 4.3 Segmentation Engine
**Location**: `pathoai/segmentation/`
**Responsibility**: Pixel-level semantic segmentation of tissue into pathologically meaningful classes.

**Tissue Classes** (following TIGER dataset annotation schema):

| Class ID | Class Name | Clinical Meaning |
|----------|------------|------------------|
| 0 | background | Non-tissue, glass, pen marks |
| 1 | tumor_bulk | Invasive breast carcinoma cells |
| 2 | stroma | Connective tissue surrounding tumor |
| 3 | lymphocytes | Immune cell infiltrates (key for sTIL) |
| 4 | necrosis | Dead/necrotic tissue regions |
| 5 | other | Blood vessels, adipose, artifacts |

**Architecture**:
- **Encoder**: EfficientNet-B3 (pretrained ImageNet; efficient-net family chosen for parameter efficiency vs. accuracy)
- **Decoder**: DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP)
- **Loss**: Weighted combination of Dice Loss (handles class imbalance) + Focal Loss (handles hard examples)
- **Input**: 512×512 RGB patches at 20× magnification
- **Output**: (H, W, 6) softmax probability maps

**Rationale for DeepLabV3+**:
- ASPP captures multi-scale context critical for tissue microarchitecture
- Skip connections from encoder preserve fine-grained boundary information
- State-of-the-art in medical image segmentation benchmarks at time of design

---

### 4.4 Detection Engine
**Location**: `pathoai/detection/`
**Responsibility**: Instance-level detection and classification of individual cells within stained tissue.

**Detected Classes**:

| Class ID | Class Name | Clinical Meaning |
|----------|------------|------------------|
| 0 | background | No cell |
| 1 | cancer_cell | Malignant epithelial cell |
| 2 | lymphocyte | TIL — counted for sTIL numerator |
| 3 | other_cell | Macrophage, fibroblast, endothelial |

**Architecture**:
- **Backbone**: ResNet34 (smaller than ResNet50; chosen for speed on limited hardware)
- **Neck**: Feature Pyramid Network (FPN) for multi-scale detection
- **Head**: Faster R-CNN region proposal + classification head
- **Input**: 256×256 RGB patches at 40× magnification (higher magnification for cell-level detail)
- **Output**: Bounding boxes (N, 4), class labels (N,), confidence scores (N,)

**Rationale for Faster R-CNN**:
- Two-stage detector provides better precision for small dense objects (lymphocytes ~8-12μm diameter)
- Established benchmark baseline in computational pathology cell detection literature
- Interpretable intermediate RPN output for debugging

---

### 4.5 Spatial Fusion Engine
**Location**: `pathoai/fusion/`
**Responsibility**: Combining segmentation masks with cell detection results to compute the sTIL score.

**sTIL Computation Methodology** (Salgado et al., 2015):

```
sTIL (%) = (N_lymphocytes_in_stroma / Total_stroma_area_mm²) × scaling_factor
```

Implementation details:
1. For each patch: project cell centroids onto coordinate system
2. Query segmentation mask at each centroid location
3. Count lymphocytes whose centroid falls in stroma class (class_id=2)
4. Compute stroma area in μm² using MPP metadata
5. Aggregate across patches using weighted mean (weight = stroma area per patch)
6. Apply clinical score discretization if configured (0-10%, 10-20%, etc.)

---

### 4.6 Validation Engine
**Location**: `pathoai/validation/`
**Responsibility**: Statistical quality control and agreement analysis.

**Operations**:
- Bootstrap 95% confidence interval (n=1000 resamples by default)
- Cohen's κ agreement with pathologist labels (when available)
- Shapiro-Wilk normality test on score distributions
- Grubbs outlier detection
- Calibration analysis (score vs. ground truth correlation)

---

### 4.7 Visualization Engine
**Location**: `pathoai/visualization/`
**Responsibility**: Publication-quality figures for papers, reports, and debugging.

**Outputs**:
- `slide_overview.png`: Thumbnail + tissue mask overlay
- `segmentation_overlay.png`: Per-patch class colormap overlaid on H&E
- `detection_overlay.png`: Bounding boxes per class color-coded
- `stil_heatmap.png`: Spatial sTIL density heatmap at slide level
- `summary_figure.png`: Multi-panel publication figure

---

### 4.8 Report Engine
**Location**: `pathoai/report/`
**Responsibility**: Structured output generation for downstream consumption.

**Formats**:
- `result.json`: Full provenance, scores, model versions, configs
- `report.pdf`: Human-readable clinical summary (future)
- `batch_results.csv`: Cohort-level aggregation

---

## 5. Data Flow Architecture

### Primary Data Contract Schemas

```python
# pathoai/core/types.py — Milestone 1 definitions

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class WSIMetadata:
    """Immutable metadata extracted from a WSI file on open."""
    slide_id: str               # Unique identifier (filename stem)
    file_path: Path             # Absolute path to WSI file
    format: str                 # File extension: .svs, .ndpi, .tif, etc.
    width: int                  # Width in pixels at level 0
    height: int                 # Height in pixels at level 0
    mpp_x: float                # Microns per pixel, x-axis at level 0
    mpp_y: float                # Microns per pixel, y-axis at level 0
    objective_power: float      # Nominal scanning magnification (e.g., 40.0)
    n_levels: int               # Number of pyramid levels
    level_dimensions: List[Tuple[int, int]]  # (W, H) per level
    level_downsamples: List[float]           # Downsample factor per level
    vendor: str                 # Scanner vendor string
    sha256: str                 # File checksum for provenance


@dataclass
class PatchCoordinateMap:
    """Bidirectional mapping between patch indices and slide coordinates."""
    slide_id: str
    level: int                         # Pyramid level patches extracted from
    patch_size_px: int                 # Size in pixels (square)
    stride_px: int                     # Stride in pixels
    mpp: float                         # MPP at extraction level
    coordinates: List[Tuple[int, int]] # (x, y) at level 0 for each patch
    tissue_flags: List[bool]           # True if patch is tissue (not background)
    n_patches_total: int
    n_patches_tissue: int


@dataclass
class CellDetectionResult:
    """Detected cells in a single patch after NMS."""
    patch_id: str
    slide_x: int               # Patch origin x at level 0
    slide_y: int               # Patch origin y at level 0
    boxes_px: np.ndarray       # Shape: (N, 4) [x1, y1, x2, y2] patch coords
    labels: np.ndarray         # Shape: (N,) integer class IDs
    scores: np.ndarray         # Shape: (N,) confidence [0, 1]
    class_names: List[str]     # Index → name mapping


@dataclass
class sTILResult:
    """Final sTIL scoring result for a single WSI."""
    slide_id: str
    stil_score_pct: float              # 0–100
    n_lymphocytes_in_stroma: int
    stroma_area_um2: float
    tumor_area_um2: float
    patch_stil_scores: np.ndarray      # Per-tissue-patch sTIL values
    ci_lower: float                    # Bootstrap 95% CI lower bound
    ci_upper: float                    # Bootstrap 95% CI upper bound
    processing_time_s: float
    seg_model_version: str
    det_model_version: str
    config_hash: str
    created_at: datetime = field(default_factory=datetime.now)
```

---

## 6. Module Dependency Graph

```
pathoai.core              (leaf — no internal pathoai deps)
    ^
    |
pathoai.wsi               (depends on: core)
    ^
    |
pathoai.segmentation      (depends on: core, wsi)
pathoai.detection         (depends on: core, wsi)
    ^                ^
    |                |
pathoai.fusion        ----+  (depends on: core, segmentation, detection)
    ^
    |
pathoai.validation        (depends on: core, fusion)
    ^
    |
pathoai.visualization     (depends on: core, wsi, segmentation, detection, fusion)
pathoai.report            (depends on: core, fusion, validation)
    ^                ^
    |                |
pathoai.dashboard    -----+  (depends on: all above — future milestone)
```

**Architectural Rule**: Dependencies flow strictly downward (from consumers to providers).
Circular imports are prohibited. Any shared type must be defined in `pathoai.core.types`.

---

## 7. Inter-Engine Communication Protocol

Engines communicate through **three mechanisms only**:

### Mechanism 1: Python Dataclass Objects
In-memory passing of typed data contracts defined in `pathoai.core.types`.

### Mechanism 2: Filesystem Artifacts
For cross-session or large data:
- Numpy arrays: `.npy` or `.npz` compressed
- Metadata: `.json` with ISO timestamp provenance
- Configs: `.yaml` versioned with experiment hash

### Mechanism 3: PyTorch Dataset Interface
For batch processing between WSI engine and model engines:
- `PatchDataset(torch.utils.data.Dataset)` wraps coordinate map into batches
- Enables `DataLoader` with multi-worker prefetching

**Rule**: Engines never call each other's internal methods. Only `PipelineOrchestrator`
(in `pathoai.core.pipeline`) coordinates cross-engine calls.

---

## 8. Configuration Architecture

### Hierarchy (lowest to highest priority):
```
config/base.yaml                 ← Defaults for all experiments
    ↓ merged with
config/datasets/tiger.yaml       ← Dataset-specific overrides
    ↓ merged with
config/models/deeplabv3plus.yaml ← Model-specific overrides
    ↓ merged with
config/experiments/<exp>.yaml    ← Experiment-specific overrides
    ↓ overridden by
Environment variables            ← Runtime overrides (PATHOAI_DEVICE=cpu)
    ↓ overridden by
CLI arguments                    ← Per-run overrides
```

### Config Access Pattern:
```python
# Always access via the singleton Config object — never parse YAML directly
from pathoai.core.config import get_config

cfg = get_config()
patch_size = cfg.wsi.patch_extraction.patch_size  # type-safe attribute access
```

---

## 9. Logging and Observability Architecture

### Log Levels:
| Level | Usage |
|-------|-------|
| `DEBUG` | Per-patch processing, tensor shapes, coordinate math |
| `INFO` | Engine milestones, slide processing start/end, scores |
| `WARNING` | Low tissue ratio, high background, non-fatal quality issues |
| `ERROR` | Recoverable failures; slide skipped with reason logged |
| `CRITICAL` | Pipeline-halting failures; exits with non-zero code |

### Log Format:
```
2026-07-18T17:09:45+05:30 [INFO ] [wsi_engine ] [exp=exp_001] [slide=TIGER-001] Extracted 2847 patches (0.23s/patch avg)
2026-07-18T17:09:45+05:30 [WARN ] [wsi_engine ] [exp=exp_001] [slide=TIGER-002] Low tissue ratio: 0.03 (threshold: 0.10) — slide skipped
```

### Observability Stack:
- **TensorBoard**: Training loss curves, validation metrics, learning rate schedules
- **Per-experiment logs**: `logs/<exp_id>/` with rotation at 50 MB
- **JSON structured logs**: Machine-parseable for automated monitoring

---

## 10. Error Handling Strategy

### Exception Hierarchy:
```
PathoAIException (base)
├── ConfigurationError         ← Invalid config schema or missing required key
├── EnvironmentError           ← Missing dependency, wrong Python version
├── DataError
│   ├── WSIReadError           ← File not found, unsupported format, corrupt file
│   ├── PatchExtractionError   ← Coordinate out of bounds, memory error
│   └── DatasetValidationError ← Missing slides, annotation mismatch
├── ModelError
│   ├── CheckpointLoadError    ← Wrong architecture, missing keys
│   ├── InferenceError         ← CUDA OOM, NaN outputs
│   └── ArchitectureError      ← Config/model mismatch
├── FusionError                ← Coordinate system mismatch
├── ValidationError            ← Statistical test failure
└── ReportGenerationError      ← File write error, template missing
```

### Rule: Never catch bare `Exception` in production code. Always catch the most specific exception available.

---

## 11. Testing Architecture

```
tests/
├── unit/                         # Fast, isolated, no GPU, no real WSIs
│   ├── core/                     # Config loading, logging, utils, validators
│   ├── wsi/                      # Reader mock, tissue detection algorithm, coord math
│   ├── segmentation/             # Model forward pass (synthetic input), loss functions
│   ├── detection/                # Model forward pass, NMS, box coordinate transforms
│   ├── fusion/                   # sTIL equation with synthetic masks/detections
│   └── validation/               # Statistical functions
├── integration/                  # Cross-engine flows (may need GPU)
│   ├── test_wsi_to_patches.py
│   ├── test_seg_pipeline.py
│   └── test_full_pipeline.py
├── fixtures/
│   ├── synthetic_wsi.py          # Generates fake .tif WSIs for CI
│   ├── synthetic_masks.py        # Generates fake segmentation masks
│   ├── synthetic_detections.py   # Generates fake cell detection outputs
│   └── configs/                  # Minimal test configs
└── conftest.py                   # Shared fixtures, test DB setup
```

**CI Strategy**: Unit tests run on every commit (CPU only, synthetic data, < 2 min). Integration tests run nightly.

---

## 12. Extensibility Design

### Model Registry Pattern:
```python
@model_registry.register("deeplabv3plus_efficientnet_b3")
class DeepLabV3PlusEffNetB3(BaseSegmentationModel):
    """Registered by string key — loaded dynamically from config."""
```

### Dataset Adapter Pattern:
```python
class TIGERDatasetAdapter(BaseDatasetAdapter):
    """Implements standard interface for TIGER dataset."""
    def list_slides(self) -> List[str]: ...
    def load_slide(self, slide_id: str) -> WSIMetadata: ...
    def load_annotations(self, slide_id: str) -> Dict: ...
```

### Stain Normalization Plugin:
```python
class MacenkoNormalizer(BaseStainNormalizer):
    """Macenko et al. (2009) stain normalization."""

class VahadaneNormalizer(BaseStainNormalizer):
    """Vahadane et al. (2016) structure-preserving normalization."""
```

New models, datasets, and normalizers can be added without modifying existing code — only new files and registry entries.

---

## 13. Deployment Architecture

### Tier 1: Research Mode (Current — Milestone 1)
- Single Windows workstation (Intel iGPU / CPU fallback)
- CLI-driven batch processing
- Results to local filesystem + TensorBoard

### Tier 2: Cloud Scale-Out (Planned — Milestone 8)
- AWS S3 for WSI storage (potentially 10s of TB)
- AWS SageMaker for distributed training
- AWS RDS for structured result storage
- AWS CLI integration (already installed on dev machine)

### Tier 3: Clinical Integration (Future)
- DICOM WADO-RS WSI ingestion
- HL7 FHIR output for EHR integration
- CE/FDA-compliant audit logging
- Hospital network security compliance

---

## 14. Security and Data Privacy

| Concern | Implementation |
|---------|---------------|
| Patient data in logs | Slide IDs hashed (SHA-256); no PHI in any log entry |
| File integrity | SHA-256 checksum verified on open; stored in result provenance |
| Reproducibility audit | Config hash + model checkpoint hash stored with every result |
| Offline operation | Pipeline runs 100% offline after initial model download |
| Access control | File-system ACL; no web-facing ports in research mode |

---

## 15. Performance Design Targets

| Operation | Target | Hardware |
|-----------|--------|----------|
| WSI thumbnail generation | < 2s | CPU |
| Tissue detection (per slide) | < 5s | CPU |
| Patch extraction (1000 patches) | < 30s | CPU |
| Segmentation inference (per patch) | < 50ms | GPU (< 300ms CPU) |
| Detection inference (per patch) | < 30ms | GPU (< 200ms CPU) |
| Spatial fusion (per slide) | < 10s | CPU |
| Full pipeline (avg. WSI, ~500 patches) | < 10 min | GPU; < 45 min CPU |
| Peak RAM usage | < 8 GB | — |
| GPU VRAM usage | < 6 GB | — |

*Note: Development machine has Intel iGPU only (no dedicated CUDA GPU). CPU inference paths are first-class, not afterthoughts.*

---

## 16. Known Limitations and Future Work

| Limitation | Impact | Mitigation | Target Milestone |
|------------|--------|-----------|------------------|
| Intel iGPU only — no NVIDIA CUDA | Slow inference | CPU PyTorch path; design CUDA-ready code | M2 |
| No OpenSlide binaries yet | Cannot read WSIs | Installation documented; scripts provided | M1 |
| No pretrained model weights | No inference | Architecture defined; training in M4/M5 | M4 |
| No real WSI data yet | No end-to-end test | Synthetic data generators | M1 |
| Windows-only dev environment | Linux deployment needed | Docker containerization planned | M8 |
| Single-machine only | Limited scale | AWS integration planned | M8 |

---

*End of Architecture Document v1.0.0*
*Next: See Pipeline.md for stage-by-stage processing specification*
