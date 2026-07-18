"""
pathoai/core/constants.py
=========================
Global named constants for the PathoAI-Platform.

ALL magic numbers used anywhere in the codebase must be defined here as
named constants. Direct use of numeric literals in engine code is forbidden
(see Implementation_Bible.md, Section 6).

Rules:
- Constants are SCREAMING_SNAKE_CASE
- Each constant has a descriptive comment explaining its scientific meaning
- Units are always stated in the comment
- Class IDs align with the TIGER dataset annotation schema

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

from typing import Dict, Tuple


# ===========================================================================
# TISSUE SEGMENTATION CLASS DEFINITIONS
# Aligns with TIGER dataset annotation schema
# Reference: TIGER Grand Challenge (https://tiger.grand-challenge.org/)
# ===========================================================================

TISSUE_CLASSES: Dict[int, str] = {
    0: "background",     # Non-tissue: glass, pen marks, artifacts, out-of-focus
    1: "tumor_bulk",     # Invasive carcinoma cells (epithelial compartment)
    2: "stroma",         # Connective tissue surrounding tumor (sTIL denominator)
    3: "lymphocytes",    # Mononuclear inflammatory cells (sTIL numerator — dense aggregates)
    4: "necrosis",       # Coagulative necrosis, ghost cells, nuclear debris
    5: "other",          # Adipose, blood vessels, DCIS, normal glands, muscle
}

# Inverse mapping: name → class ID
TISSUE_CLASS_IDS: Dict[str, int] = {v: k for k, v in TISSUE_CLASSES.items()}

# Number of tissue classes (used to define model output channels)
N_TISSUE_CLASSES: int = len(TISSUE_CLASSES)

# Class IDs for quick access (avoids magic numbers in code)
BACKGROUND_CLASS_ID: int = 0
TUMOR_CLASS_ID: int = 1
STROMA_CLASS_ID: int = 2      # sTIL denominator compartment
LYMPHOCYTE_SEG_CLASS_ID: int = 3  # Dense lymphocytic regions in segmentation
NECROSIS_CLASS_ID: int = 4
OTHER_CLASS_ID: int = 5

# ===========================================================================
# CELL DETECTION CLASS DEFINITIONS
# Used by Faster R-CNN detection head
# ===========================================================================

CELL_CLASSES: Dict[int, str] = {
    0: "background",    # No cell / non-cell region
    1: "cancer_cell",   # Malignant epithelial cell
    2: "lymphocyte",    # TIL — used in sTIL numerator (individual cell count)
    3: "other_cell",    # Macrophage, fibroblast, endothelial cell
}

CELL_CLASS_IDS: Dict[str, int] = {v: k for k, v in CELL_CLASSES.items()}

N_CELL_CLASSES: int = len(CELL_CLASSES)

CELL_BACKGROUND_ID: int = 0
CANCER_CELL_CLASS_ID: int = 1
LYMPHOCYTE_DET_CLASS_ID: int = 2    # Lymphocyte class in DETECTION (sTIL numerator)
OTHER_CELL_CLASS_ID: int = 3

# ===========================================================================
# VISUALIZATION COLOR MAPS
# RGB tuples (0-255) for class overlays in figures
# Standardized colors for reproducible figures across publications
# ===========================================================================

TISSUE_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (220, 220, 220),   # background   → light gray
    1: (255,  80,  80),   # tumor_bulk   → red
    2: ( 80, 200, 120),   # stroma       → emerald green
    3: ( 80, 120, 255),   # lymphocytes  → blue
    4: (200, 160,  80),   # necrosis     → amber
    5: (180, 180, 180),   # other        → medium gray
}

CELL_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (220, 220, 220),   # background   → light gray
    1: (255,  80,  80),   # cancer_cell  → red
    2: ( 80, 120, 255),   # lymphocyte   → blue
    3: (255, 200,  80),   # other_cell   → yellow
}

# Overlay alpha for segmentation visualization (0=transparent, 1=opaque)
SEGMENTATION_OVERLAY_ALPHA: float = 0.45

# ===========================================================================
# SPATIAL / OPTICAL CONSTANTS
# ===========================================================================

# Target microns-per-pixel for segmentation (≈ 20× magnification)
# At 20× on a typical scanner: MPP ≈ 0.50 μm/pixel
SEGMENTATION_TARGET_MPP: float = 0.50  # μm/pixel

# Target MPP for cell detection (≈ 40× magnification)
# At 40× on a typical scanner: MPP ≈ 0.25 μm/pixel
DETECTION_TARGET_MPP: float = 0.25  # μm/pixel

# Typical MPP range for 40× scans (used for validation)
MPP_40X_MIN: float = 0.15   # μm/pixel — unusually small but valid
MPP_40X_MAX: float = 0.35   # μm/pixel — unusually large but valid

# Micron conversion factor
UM2_TO_MM2: float = 1e-6   # 1 μm² = 1e-6 mm²

# Typical diameter of a lymphocyte at 40× (pixels)
# At 40× with 0.25 μm/px: lymphocyte ~10 μm diameter = ~40 pixels
LYMPHOCYTE_DIAMETER_UM: float = 10.0      # micrometers
LYMPHOCYTE_MAX_DIAMETER_UM: float = 15.0  # micrometers (upper bound)

# Typical diameter of a cancer cell at 40× (pixels)
CANCER_CELL_DIAMETER_UM: float = 20.0     # micrometers
CANCER_CELL_MAX_DIAMETER_UM: float = 40.0 # micrometers

# ===========================================================================
# PATCH EXTRACTION CONSTANTS
# ===========================================================================

# Default patch size (pixels) at extraction level
DEFAULT_PATCH_SIZE: int = 512

# Default extraction stride (pixels) — 50% overlap by default
DEFAULT_PATCH_STRIDE: int = 256

# Maximum thumbnail dimension (longest side) for tissue detection
THUMBNAIL_MAX_DIM: int = 2048

# Minimum tissue coverage ratio in a patch to include it
# Patches with < MIN_TISSUE_COVERAGE fraction of tissue are discarded
MIN_TISSUE_COVERAGE_RATIO: float = 0.25

# Overall slide tissue ratio below which slide is rejected
MIN_SLIDE_TISSUE_RATIO: float = 0.05

# Morphological kernel size for tissue mask cleanup (pixels at thumbnail)
TISSUE_MORPH_KERNEL_SIZE: int = 15

# Minimum connected component size to keep in tissue mask (pixels at thumbnail)
TISSUE_MIN_COMPONENT_PX: int = 1000

# Blank patch detection: patches with mean RGB > this threshold are discarded
BLANK_PATCH_THRESHOLD: int = 230

# Blur detection: patches with Laplacian variance < this are considered blurry
BLUR_LAPLACIAN_THRESHOLD: float = 50.0

# ===========================================================================
# FUSION / sTIL CONSTANTS
# ===========================================================================

# Minimum stroma area to compute a valid sTIL score
# Below this threshold, the score is flagged as LOW_CONFIDENCE
MIN_STROMA_AREA_MM2: float = 0.5      # mm²
MIN_STROMA_AREA_UM2: float = 0.5e6   # μm²

# Minimum number of detected lymphocytes for a reliable count
MIN_LYMPHOCYTES_FOR_CONFIDENCE: int = 50

# Cell detection confidence threshold (predictions below this are dropped)
DEFAULT_DETECTION_CONFIDENCE: float = 0.50

# Non-maximum suppression IoU threshold for cell detection
DEFAULT_NMS_IOU_THRESHOLD: float = 0.50

# ===========================================================================
# STATISTICAL VALIDATION CONSTANTS
# ===========================================================================

# Number of bootstrap resamples for confidence interval estimation
BOOTSTRAP_N_RESAMPLES: int = 1000

# Default confidence level for bootstrap CI (95%)
DEFAULT_CONFIDENCE_LEVEL: float = 0.95

# sTIL score range (clinical scale)
STIL_SCORE_MIN: float = 0.0
STIL_SCORE_MAX: float = 100.0

# Clinical cutoffs for sTIL score interpretation (TILs Working Group)
STIL_CUTOFF_LOW: float = 10.0     # < 10%: TIL-low
STIL_CUTOFF_MODERATE: float = 50.0  # 10–50%: TIL-moderate
# >= 50%: TIL-high (lymphocyte-predominant breast cancer, LPBC)

# ===========================================================================
# SUPPORTED FILE FORMATS
# ===========================================================================

SUPPORTED_WSI_FORMATS = frozenset({
    ".svs",    # Aperio (Leica) — most common in TCGA
    ".ndpi",   # Hamamatsu
    ".scn",    # Leica SCN
    ".mrxs",   # 3DHISTECH Panoramic
    ".tif",    # Generic pyramidal TIFF (TIGER dataset format)
    ".tiff",   # Alternative TIFF extension
    ".vms",    # Hamamatsu VMS
    ".vmu",    # Hamamatsu VMU
    ".bif",    # Ventana (Roche)
})

# ===========================================================================
# NORMALIZATION CONSTANTS
# ImageNet mean and std for normalizing pretrained model inputs
# ===========================================================================

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Macenko stain normalization reference stain matrix
# Derived from TIGER training subset representative slide
# Rows: [Hematoxylin vector, Eosin vector] in OD (optical density) space
# Shape: (2, 3) — 2 stains × 3 color channels (RGB)
MACENKO_REFERENCE_STAIN_MATRIX: Tuple[Tuple[float, ...], ...] = (
    (0.5626, 0.7201, 0.4062),   # Hematoxylin
    (0.2159, 0.8012, 0.5581),   # Eosin
)

# ===========================================================================
# ENVIRONMENT CONSTANTS
# ===========================================================================

# Minimum free disk space required to run pipeline (GB)
MIN_FREE_DISK_GB: float = 5.0

# Minimum RAM required (GB)
MIN_RAM_GB: float = 8.0

# Minimum Python version
MIN_PYTHON_VERSION: Tuple[int, int] = (3, 10)

# Required Python version (exact minor for reproducibility)
TARGET_PYTHON_VERSION: Tuple[int, int] = (3, 11)
