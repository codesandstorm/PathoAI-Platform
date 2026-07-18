# PathoAI-Platform: Dataset Specification

> **Document Version**: 1.0.0
> **Date**: 2026-07-18
> **Status**: Milestone 1 — Infrastructure Design
> **Primary Dataset**: TIGER (Tumor InfiltratinG lymphocytes in breast cancER)

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [TIGER Dataset Specification](#2-tiger-dataset-specification)
3. [Annotation Schema](#3-annotation-schema)
4. [Data Organization](#4-data-organization)
5. [Data Splits](#5-data-splits)
6. [Data Download Instructions](#6-data-download-instructions)
7. [Data Validation Protocol](#7-data-validation-protocol)
8. [Class Definitions](#8-class-definitions)
9. [Quality Criteria](#9-quality-criteria)
10. [Reference Datasets and Benchmarks](#10-reference-datasets-and-benchmarks)
11. [Data Augmentation Specification](#11-data-augmentation-specification)
12. [Dataset Statistics](#12-dataset-statistics)
13. [Ethical and Legal Considerations](#13-ethical-and-legal-considerations)

---

## 1. Dataset Overview

### Primary Dataset: TIGER

| Property | Value |
|----------|-------|
| **Full Name** | Tumor InfiltratinG lymphocytes in breast cancER |
| **Challenge URL** | https://tiger.grand-challenge.org/ |
| **Data Type** | Whole Slide Images (WSIs) |
| **Stain** | Hematoxylin & Eosin (H&E) |
| **Cancer Type** | Invasive Breast Carcinoma |
| **Format** | TIFF (pyramidal) |
| **Annotation Types** | Semantic segmentation masks + sTIL scores |
| **License** | CC BY-NC 4.0 (Non-commercial research) |
| **Scanner** | Multiple (scanner-agnostic by design) |
| **Approx. Total Size** | ~500 GB (full dataset) |

### Why TIGER?

The TIGER dataset is chosen as the primary training and evaluation dataset for PathoAI-Platform because:

1. **Public availability**: Fully downloadable without institutional data transfer agreements
2. **Matched task**: Specifically designed for sTIL scoring in breast cancer — directly maps to our pipeline objective
3. **Expert annotations**: Pixel-level annotations by experienced pathologists following TILs Working Group guidelines
4. **Multi-scanner diversity**: Images from multiple scanning platforms improve generalization
5. **Challenge benchmark**: Results can be directly compared to published challenge leaderboard entries
6. **Active community**: Ongoing challenge with regular updates and competitor analyses

---

## 2. TIGER Dataset Specification

### 2.1 Image Specifications

| Property | Specification |
|----------|--------------|
| **Format** | Pyramidal TIFF (.tif) |
| **Color Space** | RGB (H&E staining) |
| **Scanning Magnification** | 40× (0.226 μm/pixel nominal) |
| **Pyramid Levels** | Typically 5–8 levels |
| **Level 0 Resolution** | Variable (typically 40,000–80,000 pixels in longest dimension) |
| **Level 0 MPP (typical)** | 0.226–0.252 μm/pixel |
| **Compression** | LZW or JPEG compressed tiles |
| **Tile Size** | 256×256 or 512×512 pixels |
| **Color Depth** | 8-bit per channel (24-bit RGB) |

### 2.2 Annotation Specifications

TIGER provides two types of annotations:

#### Type A: Tissue Segmentation Annotations
- **Format**: Pyramidal TIFF mask (same resolution as WSI)
- **Encoding**: Integer class labels per pixel
- **Coordinate system**: Level 0 (highest resolution)

#### Type B: sTIL Score Annotations
- **Format**: CSV file per slide
- **Values**: Integer percentage 0–100
- **Annotators**: Multiple expert pathologists
- **Inter-rater agreement**: Published in TIGER challenge papers

### 2.3 Sub-tasks Defined by TIGER Challenge

| Sub-task | Input | Output | PathoAI Stage |
|----------|-------|--------|---------------|
| Task 1a | WSI | Tissue segmentation mask | Stage 7: Segmentation |
| Task 1b | WSI + seg mask | Cell detection (bounding boxes) | Stage 8: Detection |
| Task 2 | WSI | sTIL score (0-100%) | Stage 9: Fusion |

PathoAI-Platform is designed to implement the complete pipeline from Task 1a → 1b → 2.

---

## 3. Annotation Schema

### 3.1 Tissue Segmentation Classes

```python
# pathoai/core/constants.py — class definitions matching TIGER annotation schema

TISSUE_CLASSES = {
    0: "background",      # Non-tissue: glass, pen marks, artifacts
    1: "tumor_bulk",      # Invasive cancer cells (epithelial)
    2: "stroma",          # Surrounding connective tissue (key for sTIL denominator)
    3: "lymphocytes",     # Mononuclear inflammatory cells (key for sTIL numerator)
    4: "necrosis",        # Dead/necrotic tissue
    5: "other",           # Blood vessels, adipose tissue, debris
}

TISSUE_CLASS_COLORS = {
    0: (220, 220, 220),   # background   → light gray
    1: (255,  80,  80),   # tumor_bulk   → red
    2: ( 80, 200, 120),   # stroma       → green
    3: ( 80, 120, 255),   # lymphocytes  → blue
    4: (200, 160,  80),   # necrosis     → amber
    5: (180, 180, 180),   # other        → gray
}
```

### 3.2 Cell Detection Classes

```python
CELL_CLASSES = {
    0: "background",      # No cell detected
    1: "cancer_cell",     # Malignant epithelial cell
    2: "lymphocyte",      # TIL — used in sTIL numerator
    3: "other_cell",      # Macrophage, fibroblast, endothelial cell
}
```

### 3.3 Clinical Class Definitions (Pathology Reference)

**Stroma** (class 2):
- Fibrous connective tissue surrounding the tumor
- Contains a mixture of fibroblasts, blood vessels, and immune cells
- This compartment is the denominator region for sTIL scoring
- Must be distinguished from intratumoral (within tumor nests) regions

**Lymphocytes** (class 3 in segmentation; class 2 in detection):
- Small, round cells with dark purple nuclei and scant cytoplasm in H&E
- Diameter: 8–12 μm (approximately 35–50 pixels at 40× with 0.25 μm/pixel)
- Include T cells, B cells, and plasma cells per TILs Working Group definition
- Plasma cells are counted as TILs (per consensus recommendation)

**Tumor Bulk** (class 1):
- Sheets, nests, or cords of invasive carcinoma cells
- Larger than lymphocytes; irregular nuclear shapes; prominent nucleoli
- May have tubular, lobular, or mixed morphology

---

## 4. Data Organization

### 4.1 Expected Dataset Directory Structure

```
data/
├── raw/
│   └── tiger/
│       ├── train/
│       │   ├── images/
│       │   │   ├── TCGA-A1-A0SO-01Z-00-DX1.tif
│       │   │   ├── TCGA-A2-A0T2-01Z-00-DX1.tif
│       │   │   └── ...
│       │   ├── masks/
│       │   │   ├── TCGA-A1-A0SO-01Z-00-DX1.tif   # Segmentation mask
│       │   │   ├── TCGA-A2-A0T2-01Z-00-DX1.tif
│       │   │   └── ...
│       │   └── annotations/
│       │       ├── til_scores.csv                  # Pathologist sTIL scores
│       │       └── cell_detections/
│       │           ├── TCGA-A1-A0SO-01Z-00-DX1.json
│       │           └── ...
│       ├── val/
│       │   └── (same structure)
│       └── test/
│           └── (same structure — no labels for challenge test set)
├── processed/
│   └── tiger/
│       ├── patches/
│       │   ├── train/
│       │   │   ├── TCGA-A1-A0SO-01Z-00-DX1/
│       │   │   │   ├── patch_coords.json
│       │   │   │   ├── tissue_mask.npy
│       │   │   │   └── patches/           # Optional: pre-extracted patch files
│       │   │   └── ...
│       │   └── val/
│       └── stats/
│           ├── dataset_statistics.json
│           └── class_weights.json
└── external/
    └── reference/
        ├── stain_reference.tif             # Reference slide for normalization
        └── stain_matrix.npy                # Pre-computed reference stain matrix
```

### 4.2 File Naming Convention

```
{TCGA_BARCODE}-{SLIDE_UUID}.tif

Example: TCGA-A1-A0SO-01Z-00-DX1-UUID1234.tif
```

PathoAI uses the filename stem as the `slide_id` throughout the pipeline.

---

## 5. Data Splits

### Recommended Train/Val/Test Split

| Split | Slides | Purpose |
|-------|--------|---------|
| **Train** | ~150 slides | Model training (segmentation + detection) |
| **Validation** | ~50 slides | Hyperparameter tuning, early stopping |
| **Test** | ~50 slides | Final evaluation (never seen during training) |

**Split strategy**: Stratified by sTIL score quartile (0–10%, 10–25%, 25–50%, 50–100%) to ensure balanced representation across the score range.

**Cross-validation**: 5-fold cross-validation on training set for reliable performance estimation.

```python
# splits/tiger_splits.json — example format
{
  "version": "1.0",
  "strategy": "stratified_by_stil_quartile",
  "seed": 42,
  "train": ["TCGA-A1-A0SO-01Z-00-DX1", "TCGA-A2-A0T2-01Z-00-DX1", ...],
  "val": ["TCGA-B3-B0XX-01Z-00-DX1", ...],
  "test": ["TCGA-C4-C0YY-01Z-00-DX1", ...],
  "created_at": "2026-07-18T17:09:45+05:30"
}
```

---

## 6. Data Download Instructions

### Step 1: Register for TIGER Challenge

1. Visit https://tiger.grand-challenge.org/
2. Create an account at https://grand-challenge.org/
3. Join the TIGER challenge to obtain download access

### Step 2: Download via Grand Challenge API

```bash
# Install grand-challenge client
pip install grand-challenge-client

# Configure credentials
gc configure --api-token YOUR_TOKEN

# Download TIGER training set
gc download-dataset tiger-training --output-dir data/raw/tiger/train/
gc download-dataset tiger-validation --output-dir data/raw/tiger/val/
```

### Step 3: Verify Downloads

```bash
# Run the dataset validation script (provided in PathoAI)
python scripts/validate_dataset.py --dataset tiger --split train
```

### Alternative: Manual Download
If the grand-challenge client is unavailable, slides can be downloaded manually from the challenge website's download page (requires login).

### Storage Requirements

| Split | Estimated Size |
|-------|---------------|
| Train images | ~300 GB |
| Train masks | ~50 GB |
| Val images | ~100 GB |
| Val masks | ~15 GB |
| Total | ~465 GB |

**Ensure D:\ drive (462 GB free) is used as primary data storage.**

---

## 7. Data Validation Protocol

### 7.1 Per-Slide Validation Checklist

Every slide must pass the following before entering the pipeline:

```python
class SlideValidator:
    """Validates a single WSI and its annotations."""

    def validate(self, slide_id: str) -> ValidationResult:
        checks = [
            self._check_file_exists(slide_id),
            self._check_file_readable(slide_id),
            self._check_mpp_available(slide_id),
            self._check_mpp_in_range(slide_id),
            self._check_tissue_ratio(slide_id),
            self._check_mask_exists(slide_id),
            self._check_mask_dimensions_match(slide_id),
            self._check_mask_classes_valid(slide_id),
            self._check_stil_score_available(slide_id),
            self._check_stil_score_in_range(slide_id),
        ]
        return ValidationResult(checks)
```

### 7.2 Dataset-Level Validation

```bash
# Full dataset validation — run before any training
python scripts/validate_dataset.py \
    --dataset tiger \
    --split train \
    --output reports/dataset_validation_train.json
```

Output report includes:
- Per-slide pass/fail status
- Class distribution across all slides
- MPP distribution histogram
- sTIL score distribution
- Data completeness percentage

---

## 8. Class Definitions

### Clinical Justification for Each Class

#### Background (Class 0)
**Definition**: Any region that is not biologically relevant tissue.
**Includes**: Clean glass, mounting medium, coverslip edges, thick pen marks, out-of-focus regions, fold artifacts.
**Clinical relevance**: Must be excluded from all scoring computations.

#### Tumor Bulk (Class 1)
**Definition**: Regions containing invasive carcinoma cells.
**Includes**: Sheets, nests, cords, or single malignant epithelial cells.
**Excludes**: In-situ carcinoma (DCIS) — included as "other" per TIGER schema.
**Clinical relevance**: Defines the tumor boundary; separates intratumoral from stromal TILs.

#### Stroma (Class 2)
**Definition**: The fibrous connective tissue compartment surrounding tumor nests.
**Includes**: Desmoplastic stroma, fibroblasts, blood vessels within stroma, and immune cells within stroma.
**Excludes**: Fat tissue (adipose), smooth muscle.
**Clinical relevance**: This is the denominator compartment for sTIL computation. Per TILs Working Group, all stromal area (including immune cell-rich stroma) counts in the denominator.

#### Lymphocytes (Class 3 in segmentation)
**Note**: In **segmentation**, class 3 represents patches of dense lymphocytic infiltrate (used for rough estimation and visualization).
In **detection** (cell-level), lymphocytes are detected individually as bounding boxes.
**Clinical relevance**: The numerator of the sTIL score. Only stromal TILs count (intratumoral TILs are noted separately but not included in sTIL by current guidelines).

#### Necrosis (Class 4)
**Definition**: Regions of coagulative necrosis, ghost cells, nuclear debris.
**Clinical relevance**: Excluded from sTIL scoring. High necrosis is an independent prognostic factor.

#### Other (Class 5)
**Definition**: Normal breast tissue, adipose, blood vessels, DCIS, surgical margins.
**Clinical relevance**: Not scored but present in tissue. Excluded from sTIL computation.

---

## 9. Quality Criteria

### Slide-Level Quality Gates

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| Tissue area ratio | ≥ 0.05 | Skip if below |
| Focus quality (blur) | Laplacian var > 50 | Warn if below |
| Stain intensity range | Mean sat. in [30, 200] | Warn if outside |
| Slide dimensions | Both > 5000 px at level 0 | Skip if below |
| MPP available | Not None | Skip if missing |
| MPP range | 0.15–0.35 μm/px (at 40×) | Warn if outside |

### Annotation Quality Flags

| Flag | Condition |
|------|-----------|
| `SPARSE_ANNOTATION` | < 20% of tissue area annotated |
| `CLASS_IMBALANCE_EXTREME` | Any class > 95% or < 0.1% of annotated area |
| `SCORE_OUTLIER` | sTIL score > 3 SD from cohort mean |
| `MULTI_ANNOTATOR_DISAGREEMENT` | κ < 0.6 between annotators (when multi-annotator available) |

---

## 10. Reference Datasets and Benchmarks

### Additional Datasets for Evaluation (Future Milestones)

| Dataset | Description | Use |
|---------|-------------|-----|
| **TCGA-BRCA** | TCGA breast cancer WSIs (no TIL annotations) | Generalization testing |
| **BreCaHAD** | Breast cancer histology with TIL annotations | Cross-dataset validation |
| **NuCLS** | Nucleus-level classification in breast cancer | Detection head pre-training |
| **PanNuke** | Pan-cancer nuclear segmentation | Backbone pre-training |

### Published Benchmark Results (for comparison)

| Method | sTIL Pearson r | AUC (high vs. low) | Reference |
|--------|---------------|---------------------|-----------|
| Pathologist (avg.) | 1.00 (reference) | 1.00 | TIGER leaderboard |
| Top TIGER challenge method | ~0.85 | ~0.92 | Grand Challenge 2022 |
| DeepLabV3+ baseline | ~0.75 | ~0.85 | Expected baseline |
| PathoAI target | ≥ 0.80 | ≥ 0.88 | Project target |

---

## 11. Data Augmentation Specification

### Training-Time Augmentations (via Albumentations)

```python
import albumentations as A

TRAIN_TRANSFORMS = A.Compose([
    # Geometric augmentations
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),

    # Scale and elastic deformations
    A.RandomScale(scale_limit=0.15, p=0.3),
    A.ElasticTransform(alpha=120, sigma=12, p=0.2),
    A.GridDistortion(p=0.2),

    # Color augmentations (conservative — H&E specific)
    A.HueSaturationValue(
        hue_shift_limit=10,      # Small hue shifts for H&E realism
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.5
    ),
    A.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02,
        p=0.3
    ),

    # Blur and noise (simulate focus variation and sensor noise)
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.2),
    A.GaussNoise(var_limit=(10, 50), p=0.2),

    # Normalize to ImageNet mean/std (after stain normalization)
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    A.pytorch.ToTensorV2(),
])

VALIDATION_TRANSFORMS = A.Compose([
    # No geometric or color augmentation for validation — only normalization
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    A.pytorch.ToTensorV2(),
])
```

**Note on H&E-specific augmentation**: Stain augmentation is handled by the stain normalization step (or optionally by color jitter with conservative parameters). Avoid aggressive color augmentation that produces non-physiological staining patterns.

---

## 12. Dataset Statistics

### Expected TIGER Training Set Statistics (approximate)

| Statistic | Value |
|-----------|-------|
| Total slides | ~195 |
| Slides with segmentation masks | ~150 |
| Slides with sTIL scores | ~195 |
| Mean sTIL score | ~22% |
| Median sTIL score | ~15% |
| sTIL score range | 0–90% |
| Mean slide dimensions | ~50,000 × 40,000 px |
| Mean tissue area per slide | ~1,500 mm² |
| Mean stroma fraction | ~45% of tissue |
| Mean tumor fraction | ~30% of tissue |
| Class distribution (stroma) | ~45% |
| Class distribution (tumor) | ~30% |
| Class distribution (background) | ~15% |
| Class distribution (lymphocytes) | ~5% |
| Class distribution (necrosis) | ~3% |
| Class distribution (other) | ~2% |

*These are estimates; run `scripts/compute_dataset_stats.py` for exact values after download.*

---

## 13. Ethical and Legal Considerations

### Data License
TIGER dataset: **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International)
- ✅ Research use permitted
- ✅ Modification and derived works permitted
- ❌ Commercial use prohibited
- ✅ Attribution required (cite TIGER challenge paper)

### Required Citation
```bibtex
@inproceedings{tiger2022,
  title={TIGER: Tumor InfiltratinG lymphocytes in breast cancER},
  booktitle={Grand Challenge on Computational Pathology},
  year={2022},
  url={https://tiger.grand-challenge.org/}
}

@article{salgado2015,
  title={The evaluation of tumor-infiltrating lymphocytes (TILs) in breast cancer},
  author={Salgado, Roberto and others},
  journal={Annals of Oncology},
  volume={26},
  number={2},
  pages={259--271},
  year={2015}
}
```

### Patient Privacy
- All TIGER slides are de-identified per Grand Challenge data standards
- No additional de-identification is required
- Do not attempt to re-identify patients
- Do not share raw WSI files outside your research team without verifying license compliance

### Institutional Data Handling
If your institution has an IRB/ethics board:
- TIGER (publicly available, de-identified data) typically does not require additional IRB approval
- Consult your institution's research ethics board if combining TIGER with any local clinical data

---

*End of Dataset Specification v1.0.0*
*Next: See Implementation_Bible.md for coding standards and engineering conventions*
