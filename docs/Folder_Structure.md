# PathoAI-Platform: Folder Structure Specification

> **Document Version**: 1.0.0
> **Date**: 2026-07-18
> **Status**: Milestone 1 — Infrastructure Design

---

## Table of Contents

1. [Complete Directory Tree](#1-complete-directory-tree)
2. [Directory Descriptions](#2-directory-descriptions)
3. [Key File Descriptions](#3-key-file-descriptions)
4. [Naming Rules Summary](#4-naming-rules-summary)
5. [What NOT to Commit](#5-what-not-to-commit)

---

## 1. Complete Directory Tree

```
PathoAI-Platform/                           ← Project root (D:\Research\PathoAI-Platform)
│
├── README.md                               ← Project overview, quick start
├── INSTALL.md                              ← Full installation guide (OpenSlide, CUDA, etc.)
├── CHANGELOG.md                            ← Version history
├── LICENSE                                 ← MIT License
├── .gitignore                              ← Git exclusions
├── .env.example                            ← Template for environment variables
├── pyproject.toml                          ← Project metadata, build system config
├── setup.py                                ← Package installation entry point
├── requirements.txt                        ← Pinned runtime dependencies
├── requirements-dev.txt                    ← Development dependencies (pytest, ruff, etc.)
├── requirements.lock                       ← Exact pinned versions (pip freeze output)
│
├── pathoai/                                ← Main Python package
│   ├── __init__.py                         ← Package root: version, author
│   │
│   ├── core/                               ← Platform Core: cross-cutting concerns
│   │   ├── __init__.py
│   │   ├── config.py                       ← Config loader, merger, singleton
│   │   ├── logger.py                       ← Structured logger factory
│   │   ├── constants.py                    ← All named constants (class IDs, colors, limits)
│   │   ├── exceptions.py                   ← Custom exception hierarchy
│   │   ├── types.py                        ← Shared dataclasses and TypeAlias definitions
│   │   ├── validators.py                   ← Input/output validation functions
│   │   ├── registry.py                     ← Model and component registry (decorator pattern)
│   │   ├── reproducibility.py              ← Global seed setting, determinism utilities
│   │   ├── environment.py                  ← Environment audit and validation
│   │   ├── pipeline.py                     ← PipelineOrchestrator: wires all engines
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── io_utils.py                 ← File I/O: safe read/write, JSON/YAML helpers
│   │       ├── image_utils.py              ← General image utilities (resize, crop, convert)
│   │       ├── math_utils.py               ← Area computation, coordinate transforms, MPP math
│   │       ├── hash_utils.py               ← SHA-256 file hashing, config hashing
│   │       └── time_utils.py               ← Timing decorators, duration formatting
│   │
│   ├── wsi/                                ← WSI Engine
│   │   ├── __init__.py
│   │   ├── reader.py                       ← WSIReader: OpenSlide wrapper with context manager
│   │   ├── metadata.py                     ← Metadata extraction from OpenSlide properties
│   │   ├── thumbnail.py                    ← Thumbnail generation at appropriate pyramid level
│   │   ├── tissue.py                       ← TissueDetector: Otsu/HSV tissue segmentation
│   │   ├── extractor.py                    ← PatchExtractor: sliding window, tissue filtering
│   │   ├── coordinate_map.py               ← PatchCoordinateMap: index ↔ slide coordinates
│   │   ├── dataset.py                      ← PatchDataset: PyTorch Dataset wrapper
│   │   └── stain/
│   │       ├── __init__.py
│   │       ├── base.py                     ← BaseStainNormalizer abstract class
│   │       ├── macenko.py                  ← MacenkoNormalizer implementation
│   │       ├── vahadane.py                 ← VahadaneNormalizer implementation (future)
│   │       └── reinhard.py                 ← ReinhardNormalizer implementation (future)
│   │
│   ├── segmentation/                       ← Segmentation Engine
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                     ← BaseSegmentationModel abstract class
│   │   │   ├── deeplabv3plus.py            ← DeepLabV3+ implementation (Milestone 4)
│   │   │   └── registry.py                 ← Segmentation model registry
│   │   ├── losses/
│   │   │   ├── __init__.py
│   │   │   ├── dice.py                     ← Dice Loss implementation
│   │   │   ├── focal.py                    ← Focal Loss implementation
│   │   │   └── combined.py                 ← Dice + Focal combined loss
│   │   ├── trainer.py                      ← Segmentation training loop (Milestone 4)
│   │   ├── evaluator.py                    ← Segmentation evaluation metrics (IoU, Dice)
│   │   └── inference.py                    ← Batch inference + slide-level mask assembly
│   │
│   ├── detection/                          ← Detection Engine
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                     ← BaseDetectionModel abstract class
│   │   │   ├── faster_rcnn.py              ← Faster R-CNN implementation (Milestone 5)
│   │   │   └── registry.py                 ← Detection model registry
│   │   ├── losses/
│   │   │   ├── __init__.py
│   │   │   └── detection_losses.py         ← RPN + classification losses
│   │   ├── transforms.py                   ← Detection-specific augmentations
│   │   ├── trainer.py                      ← Detection training loop (Milestone 5)
│   │   ├── evaluator.py                    ← COCO mAP evaluation
│   │   └── inference.py                    ← Batch inference + coordinate normalization
│   │
│   ├── fusion/                             ← Spatial Fusion Engine
│   │   ├── __init__.py
│   │   ├── spatial_ops.py                  ← Centroid-in-mask queries, coordinate intersection
│   │   ├── stil_computer.py                ← sTIL equation implementation
│   │   ├── aggregator.py                   ← Patch-to-slide score aggregation
│   │   └── fusion_engine.py                ← High-level fusion orchestration
│   │
│   ├── validation/                         ← Validation Engine
│   │   ├── __init__.py
│   │   ├── bootstrap.py                    ← Bootstrap CI estimation
│   │   ├── agreement.py                    ← Inter-rater agreement (Cohen's κ, ICC)
│   │   ├── statistics.py                   ← Normality tests, outlier detection
│   │   ├── quality_flags.py                ← Quality flag assignment logic
│   │   └── validation_engine.py            ← High-level validation orchestration
│   │
│   ├── visualization/                      ← Visualization Engine
│   │   ├── __init__.py
│   │   ├── overlay.py                      ← Segmentation + detection overlay renderers
│   │   ├── heatmap.py                      ← sTIL spatial density heatmap
│   │   ├── figures.py                      ← Publication-quality multi-panel figures
│   │   ├── colors.py                       ← Class color maps and palettes
│   │   └── slide_view.py                   ← Slide thumbnail + tissue mask overlay
│   │
│   ├── report/                             ← Report Engine
│   │   ├── __init__.py
│   │   ├── json_reporter.py                ← JSON result generation with provenance
│   │   ├── csv_exporter.py                 ← Batch CSV export
│   │   └── pdf_reporter.py                 ← PDF clinical summary (future)
│   │
│   └── dashboard/                          ← Clinical Dashboard (Future Milestone 7+)
│       ├── __init__.py
│       ├── api/
│       │   └── __init__.py                 ← REST API stub
│       └── ui/
│           └── __init__.py                 ← Web UI stub
│
├── config/                                 ← All YAML configuration files
│   ├── base.yaml                           ← Master defaults for all settings
│   ├── datasets/
│   │   ├── tiger.yaml                      ← TIGER dataset-specific settings
│   │   └── tcga_brca.yaml                  ← TCGA BRCA settings (future)
│   ├── models/
│   │   ├── deeplabv3plus_efficientnet_b3.yaml ← Segmentation model config
│   │   └── faster_rcnn_resnet34.yaml       ← Detection model config
│   └── experiments/
│       ├── exp_001_baseline.yaml           ← First baseline experiment
│       └── exp_002_augmented.yaml          ← Augmented training experiment (future)
│
├── data/                                   ← Data directory (gitignored except structure)
│   ├── .gitkeep                            ← Preserves directory in git
│   ├── raw/
│   │   └── tiger/
│   │       ├── train/
│   │       │   ├── images/                 ← Raw .tif WSI files
│   │       │   ├── masks/                  ← Segmentation annotation masks
│   │       │   └── annotations/            ← CSV sTIL scores, JSON cell annotations
│   │       ├── val/
│   │       └── test/
│   ├── processed/
│   │   └── tiger/
│   │       ├── patches/                    ← Pre-extracted patch data
│   │       └── stats/                      ← Dataset statistics
│   └── external/
│       └── reference/
│           └── stain_reference.tif         ← Stain normalization reference slide
│
├── models/                                 ← Model checkpoints (gitignored)
│   ├── .gitkeep
│   ├── pretrained/                         ← Downloaded ImageNet pretrained weights
│   │   ├── efficientnet_b3_imagenet.pth
│   │   └── resnet34_imagenet.pth
│   └── trained/                            ← PathoAI-trained checkpoints
│       ├── segmentation/
│       │   ├── exp_001/
│       │   │   ├── best_val_dice.pth
│       │   │   └── training_log.json
│       │   └── ...
│       └── detection/
│           └── ...
│
├── results/                                ← Pipeline outputs per slide (gitignored)
│   ├── .gitkeep
│   └── tiger_train/
│       └── TCGA-A1-A0SO-01Z-00-DX1/
│           ├── preflight_manifest.json
│           ├── metadata.json
│           ├── thumbnail.png
│           ├── tissue_mask.npy
│           ├── patch_coords.json
│           ├── segmentation_map.npy
│           ├── stroma_mask.npy
│           ├── tumor_mask.npy
│           ├── detections_summary.json
│           ├── stil_result.json
│           ├── validation_summary.json
│           ├── thumbnail_tissue.png
│           ├── segmentation_overlay.png
│           ├── detection_overlay.png
│           ├── stil_heatmap.png
│           └── summary_figure.png
│
├── logs/                                   ← Log files (gitignored)
│   ├── .gitkeep
│   └── exp_001_20260718/
│       ├── pathoai.log
│       └── pathoai_structured.jsonl
│
├── tests/                                  ← Complete test suite
│   ├── __init__.py
│   ├── conftest.py                         ← Shared fixtures and pytest configuration
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── test_config.py
│   │   │   ├── test_logger.py
│   │   │   ├── test_constants.py
│   │   │   ├── test_exceptions.py
│   │   │   ├── test_validators.py
│   │   │   ├── test_registry.py
│   │   │   ├── test_reproducibility.py
│   │   │   └── utils/
│   │   │       ├── test_io_utils.py
│   │   │       ├── test_image_utils.py
│   │   │       ├── test_math_utils.py
│   │   │       └── test_hash_utils.py
│   │   ├── wsi/
│   │   │   ├── __init__.py
│   │   │   ├── test_reader.py
│   │   │   ├── test_metadata.py
│   │   │   ├── test_thumbnail.py
│   │   │   ├── test_tissue.py
│   │   │   ├── test_extractor.py
│   │   │   ├── test_coordinate_map.py
│   │   │   ├── test_dataset.py
│   │   │   └── stain/
│   │   │       └── test_macenko.py
│   │   ├── segmentation/
│   │   │   ├── __init__.py
│   │   │   ├── models/
│   │   │   │   └── test_deeplabv3plus.py
│   │   │   └── losses/
│   │   │       ├── test_dice.py
│   │   │       └── test_focal.py
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   ├── models/
│   │   │   │   └── test_faster_rcnn.py
│   │   │   └── test_transforms.py
│   │   ├── fusion/
│   │   │   ├── __init__.py
│   │   │   ├── test_spatial_ops.py
│   │   │   ├── test_stil_computer.py
│   │   │   └── test_aggregator.py
│   │   └── validation/
│   │       ├── __init__.py
│   │       ├── test_bootstrap.py
│   │       ├── test_agreement.py
│   │       └── test_statistics.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_wsi_to_patches.py          ← WSI → PatchDataset end-to-end
│   │   ├── test_seg_pipeline.py            ← PatchDataset → SegmentationMask
│   │   └── test_full_pipeline.py           ← WSI → sTIL score (with synthetic data)
│   └── fixtures/
│       ├── __init__.py
│       ├── synthetic_wsi.py                ← Synthetic WSI generator (no real files needed)
│       ├── synthetic_masks.py              ← Synthetic segmentation mask generator
│       ├── synthetic_detections.py         ← Synthetic detection result generator
│       └── configs/
│           ├── test_base.yaml              ← Minimal config for unit tests
│           └── test_wsi.yaml               ← WSI-engine-specific test config
│
├── scripts/                                ← Utility and operational scripts
│   ├── setup_environment.py               ← Automated environment setup and validation
│   ├── validate_dataset.py                ← Dataset integrity validation
│   ├── compute_dataset_stats.py           ← Compute class distributions, sTIL histograms
│   ├── download_pretrained.py             ← Download ImageNet pretrained weights
│   ├── run_pipeline.py                    ← CLI entry point for full pipeline
│   ├── run_training_seg.py                ← Segmentation training script
│   ├── run_training_det.py                ← Detection training script
│   └── generate_splits.py                 ← Create stratified train/val/test splits
│
├── notebooks/                             ← Jupyter notebooks for exploration and visualization
│   ├── 01_data_exploration.ipynb          ← TIGER dataset exploration
│   ├── 02_tissue_detection_demo.ipynb     ← Interactive tissue detection demo
│   ├── 03_segmentation_results.ipynb      ← Segmentation visualization (Milestone 4)
│   ├── 04_detection_results.ipynb         ← Detection visualization (Milestone 5)
│   ├── 05_stil_analysis.ipynb             ← sTIL score analysis and statistics
│   └── 06_validation_report.ipynb         ← Full validation report generation
│
├── docs/                                  ← All documentation
│   ├── Architecture.md                    ← System architecture specification
│   ├── Pipeline.md                        ← Stage-by-stage pipeline specification
│   ├── Dataset_Specification.md           ← TIGER dataset integration guide
│   ├── Implementation_Bible.md            ← Coding standards and conventions
│   ├── Folder_Structure.md                ← This document
│   ├── INSTALL.md                         ← Installation guide
│   ├── API_Reference.md                   ← Auto-generated API docs (future)
│   └── adr/                               ← Architecture Decision Records
│       ├── ADR-001-openslide-reader.md
│       ├── ADR-002-deeplabv3plus-segmentation.md
│       ├── ADR-003-faster-rcnn-detection.md
│       ├── ADR-004-tiger-dataset.md
│       ├── ADR-005-cpu-first-design.md
│       └── ADR-006-pytorch-over-tensorflow.md
│
├── .github/                               ← GitHub CI/CD configuration
│   └── workflows/
│       ├── ci.yml                         ← Unit test CI on every push
│       └── nightly.yml                    ← Integration test CI nightly
│
└── splits/                                ← Dataset split definitions
    ├── tiger_train_val_test.json          ← Official train/val/test slide IDs
    └── tiger_5fold_cv.json                ← 5-fold CV split definitions
```

---

## 2. Directory Descriptions

| Directory | Purpose | Gitignored? |
|-----------|---------|-------------|
| `pathoai/` | Main Python package — all source code | No |
| `pathoai/core/` | Cross-cutting: config, logging, types, utils | No |
| `pathoai/wsi/` | WSI reading and preprocessing engine | No |
| `pathoai/segmentation/` | Tissue segmentation models and training | No |
| `pathoai/detection/` | Cell detection models and training | No |
| `pathoai/fusion/` | Spatial fusion and sTIL computation | No |
| `pathoai/validation/` | Statistical validation engine | No |
| `pathoai/visualization/` | Figure and overlay generation | No |
| `pathoai/report/` | Result serialization | No |
| `config/` | YAML configuration files | No |
| `data/` | Raw and processed dataset files | **Yes** (except `.gitkeep`) |
| `models/` | Model checkpoints | **Yes** (except `.gitkeep`) |
| `results/` | Pipeline output per slide | **Yes** (except `.gitkeep`) |
| `logs/` | Log files | **Yes** (except `.gitkeep`) |
| `tests/` | Complete test suite | No |
| `scripts/` | Utility CLI scripts | No |
| `notebooks/` | Jupyter exploration notebooks | No (tracked, not `.ipynb` outputs) |
| `docs/` | All documentation | No |
| `splits/` | Dataset split JSON files | No |

---

## 3. Key File Descriptions

### Root Level

| File | Purpose |
|------|---------|
| `README.md` | Project overview, features, quick start guide |
| `INSTALL.md` | Step-by-step installation for all dependencies including OpenSlide |
| `CHANGELOG.md` | Version history with semantic versioning |
| `pyproject.toml` | Build system config (setuptools), project metadata |
| `requirements.txt` | Runtime dependencies with version bounds |
| `requirements-dev.txt` | Additional dev dependencies (ruff, pytest-cov, etc.) |
| `.env.example` | Template for environment variables (PATHOAI_DATA_DIR, etc.) |
| `.gitignore` | Excludes data/, models/, results/, logs/, __pycache__, .env |

### Core Module Files

| File | Key Classes/Functions |
|------|----------------------|
| `pathoai/core/config.py` | `ConfigManager`, `get_config()`, `load_experiment_config()` |
| `pathoai/core/logger.py` | `get_logger(name)`, `StructuredLogger`, `ExperimentLogger` |
| `pathoai/core/constants.py` | `TISSUE_CLASSES`, `CELL_CLASSES`, `STROMA_CLASS_ID`, `SEGMENTATION_TARGET_MPP`, etc. |
| `pathoai/core/exceptions.py` | `PathoAIException`, `WSIReadError`, `DataError`, `ModelError`, etc. |
| `pathoai/core/types.py` | `WSIMetadata`, `PatchCoordinateMap`, `CellDetectionResult`, `sTILResult`, etc. |
| `pathoai/core/validators.py` | `validate_wsi_metadata()`, `validate_patch_dataset()`, `validate_stil_result()` |
| `pathoai/core/registry.py` | `ModelRegistry`, `@registry.register()` decorator |
| `pathoai/core/reproducibility.py` | `set_global_seed()`, `capture_environment_snapshot()` |
| `pathoai/core/environment.py` | `EnvironmentValidator`, `validate_environment()`, `generate_env_report()` |
| `pathoai/core/pipeline.py` | `PipelineOrchestrator`, `run_pipeline()` |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/base.yaml` | Master defaults — every setting has a sensible default here |
| `config/datasets/tiger.yaml` | TIGER-specific: data paths, class mappings, split paths |
| `config/models/deeplabv3plus_efficientnet_b3.yaml` | Model architecture hyperparameters |
| `config/models/faster_rcnn_resnet34.yaml` | Detection model hyperparameters |
| `config/experiments/exp_001_baseline.yaml` | Baseline experiment overrides |

---

## 4. Naming Rules Summary

| Item | Convention | Example |
|------|-----------|---------|
| Python files | `snake_case.py` | `patch_extractor.py` |
| Test files | `test_<module>.py` | `test_patch_extractor.py` |
| Config files | `snake_case.yaml` | `tiger_dataset.yaml` |
| Documentation | `PascalCase.md` | `Architecture.md` |
| Directories | `snake_case` | `pathoai/wsi/`, `pathoai/core/utils/` |
| Python classes | `PascalCase` | `PatchExtractor`, `WSIReader` |
| Python functions | `snake_case` | `extract_patches()`, `detect_tissue()` |
| Python constants | `SCREAMING_SNAKE_CASE` | `STROMA_CLASS_ID = 2` |
| Experiment names | `exp_NNN_<descriptor>` | `exp_001_baseline` |
| ADR files | `ADR-NNN-<short-title>.md` | `ADR-001-openslide-reader.md` |
| Git branches | `<type>/<descriptor>` | `feat/wsi-tissue-detection` |
| Result directories | `<dataset>_<split>/` | `tiger_train/` |
| Log directories | `<exp_id>_<YYYYMMDD>/` | `exp_001_20260718/` |

---

## 5. What NOT to Commit

The following must NEVER be committed to git:

```gitignore
# .gitignore content

# Data — too large, not code
data/raw/
data/processed/

# Model checkpoints — too large
models/pretrained/
models/trained/

# Pipeline results — reproducible from code
results/

# Logs
logs/

# Python artifacts
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Jupyter notebook outputs (keep .ipynb files, not cell outputs)
# Use nbstripout to strip outputs before commit

# Environment
.env                    # Contains secrets/passwords
venv/
.venv/
pathoai_env/

# IDE
.vscode/settings.json  # Personal settings
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Testing artifacts
.pytest_cache/
.coverage
htmlcov/

# TensorBoard event files
runs/
tensorboard/
```

### Exception: `.gitkeep` files
Empty directories required by the project structure (data/, models/, results/, logs/) contain a `.gitkeep` file that IS committed to preserve the directory skeleton in the repository.

---

*End of Folder Structure Specification v1.0.0*
*All engines, tests, configs, and scripts follow this structure. Deviations require ADR justification.*
