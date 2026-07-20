"""
pathoai/core/types.py
=====================
Shared data contract types for PathoAI-Platform.

All inter-engine data transfer objects (DTOs) are defined here as
Python dataclasses. This ensures:
1. Type safety across engine boundaries
2. Clear documentation of what data each engine produces/consumes
3. No circular imports (all engines import from core.types, never from each other)
4. JSON serialization support via to_dict() / from_dict() methods

Coordinate system convention:
    - All slide coordinates are in level-0 pixel space (OpenSlide convention)
    - Physical coordinates (μm) are computed using MPP metadata
    - Patch-local coordinates are relative to the patch's top-left corner

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# WSI ENGINE OUTPUT TYPES
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WSIMetadata:
    """Immutable metadata extracted from a WSI file on open.

    Populated by WSIReader.get_metadata() and stored alongside results
    for full provenance tracking.

    Attributes
    ----------
    slide_id : str
        Unique identifier derived from filename stem.
        Example: "TCGA-A1-A0SO-01Z-00-DX1"
    file_path : Path
        Absolute path to the WSI file.
    format : str
        File extension including dot. Example: ".tif", ".svs"
    width : int
        Width of the slide at level 0 (highest resolution), in pixels.
    height : int
        Height of the slide at level 0 (highest resolution), in pixels.
    mpp_x : float
        Microns per pixel in the x-axis at level 0.
        Essential for converting pixel distances to physical distances.
    mpp_y : float
        Microns per pixel in the y-axis at level 0.
        Typically equal to mpp_x for most scanners.
    objective_power : float
        Nominal scanning magnification. Example: 40.0
    n_levels : int
        Number of resolution levels in the image pyramid.
    level_dimensions : List[Tuple[int, int]]
        Width × height (pixels) at each pyramid level.
        Index 0 = highest resolution (level 0).
    level_downsamples : List[float]
        Downsampling factor relative to level 0 for each level.
        level_downsamples[0] is always 1.0.
    vendor : str
        Scanner/vendor string from slide properties.
    sha256 : str
        SHA-256 hex digest of the WSI file for provenance verification.
    """

    slide_id: str
    file_path: Path
    format: str
    width: int
    height: int
    mpp_x: float
    mpp_y: float
    objective_power: float
    n_levels: int
    level_dimensions: List[Tuple[int, int]]
    level_downsamples: List[float]
    vendor: str
    sha256: str

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Width × height at level 0."""
        return (self.width, self.height)

    @property
    def mpp(self) -> float:
        """Mean MPP (assumes mpp_x ≈ mpp_y, which holds for most scanners)."""
        return (self.mpp_x + self.mpp_y) / 2.0

    @property
    def area_um2(self) -> float:
        """Total slide area at level 0 in square micrometers."""
        return float(self.width * self.height) * self.mpp_x * self.mpp_y

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dictionary."""
        d = asdict(self)
        d["file_path"] = str(d["file_path"])
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "WSIMetadata":
        """Deserialize from dictionary (e.g., loaded from JSON)."""
        d = d.copy()
        d["file_path"] = Path(d["file_path"])
        d["level_dimensions"] = [tuple(x) for x in d["level_dimensions"]]
        return cls(**d)


@dataclass
class PatchCoordinateMap:
    """Bidirectional mapping between patch indices and slide coordinates.

    Produced by PatchExtractor. Used by downstream engines to map
    patch-level results back to slide coordinates and to assemble
    slide-level output masks.

    Attributes
    ----------
    slide_id : str
        Identifier of the source slide.
    level : int
        Pyramid level from which patches were extracted.
    patch_size_px : int
        Size of each patch in pixels (square patches only).
    stride_px : int
        Stride (step size) used during sliding window extraction, in pixels.
    mpp : float
        Microns per pixel at the extraction level.
    coordinates : List[Tuple[int, int]]
        List of (x, y) coordinates at level 0 for the top-left corner of each
        patch. Length equals n_patches_total (only tissue patches included
        if filter_tissue=True during extraction).
    tissue_flags : List[bool]
        True if the corresponding patch passes tissue coverage threshold.
        Only tissue-positive patches are included in `coordinates` when
        tissue filtering is enabled.
    n_patches_total : int
        Total number of patches in the sliding window grid (before filtering).
    n_patches_tissue : int
        Number of patches that passed tissue filtering (len(coordinates)).
    """

    slide_id: str
    level: int
    patch_size_px: int
    stride_px: int
    mpp: float
    coordinates: List[Tuple[int, int]]
    tissue_flags: List[bool]
    n_patches_total: int
    n_patches_tissue: int

    def __len__(self) -> int:
        return len(self.coordinates)

    def get_slide_coordinate(self, patch_idx: int) -> Tuple[int, int]:
        """Get slide-level (level 0) coordinate for a patch index."""
        return self.coordinates[patch_idx]

    def get_patch_area_um2(self) -> float:
        """Physical area of a single patch in square micrometers."""
        return float(self.patch_size_px ** 2) * (self.mpp ** 2)

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "slide_id": self.slide_id,
            "level": self.level,
            "patch_size_px": self.patch_size_px,
            "stride_px": self.stride_px,
            "mpp": self.mpp,
            "coordinates": self.coordinates,
            "tissue_flags": self.tissue_flags,
            "n_patches_total": self.n_patches_total,
            "n_patches_tissue": self.n_patches_tissue,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PatchCoordinateMap":
        """Deserialize from dictionary."""
        d = d.copy()
        d["coordinates"] = [tuple(c) for c in d["coordinates"]]
        return cls(**d)

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PatchCoordinateMap":
        """Load from JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# SEGMENTATION ENGINE OUTPUT TYPES
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """Pixel-wise segmentation output for a single patch.

    Attributes
    ----------
    patch_id : str
        Unique identifier for this patch (e.g., "slide001_patch_0042").
    slide_x : int
        X coordinate of patch top-left corner at slide level 0.
    slide_y : int
        Y coordinate of patch top-left corner at slide level 0.
    class_probs : np.ndarray
        Softmax class probabilities. Shape: (H, W, n_classes), float32.
    predicted_classes : np.ndarray
        Argmax class prediction per pixel. Shape: (H, W), uint8.
    mpp : float
        Microns per pixel at the extraction level.
    """

    patch_id: str
    slide_x: int
    slide_y: int
    class_probs: np.ndarray    # (H, W, n_classes), float32
    predicted_classes: np.ndarray  # (H, W), uint8
    mpp: float

    def get_binary_mask(self, class_id: int) -> np.ndarray:
        """Return binary mask for a specific tissue class.

        Parameters
        ----------
        class_id : int
            The tissue class ID (see constants.TISSUE_CLASSES).

        Returns
        -------
        np.ndarray
            Boolean mask, shape (H, W). True where predicted_classes == class_id.
        """
        return self.predicted_classes == class_id


# ---------------------------------------------------------------------------
# DETECTION ENGINE OUTPUT TYPES
# ---------------------------------------------------------------------------

@dataclass
class CellDetectionResult:
    """Detected cells in a single patch after NMS and confidence filtering.

    Attributes
    ----------
    patch_id : str
        Unique identifier for the source patch.
    slide_x : int
        X coordinate of patch top-left corner at slide level 0.
    slide_y : int
        Y coordinate of patch top-left corner at slide level 0.
    boxes_px : np.ndarray
        Bounding boxes in patch-local pixel coordinates.
        Shape: (N, 4), float32. Format: [x1, y1, x2, y2].
    labels : np.ndarray
        Integer class ID for each detected cell.
        Shape: (N,), int64.
    scores : np.ndarray
        Detection confidence score for each cell. Range [0, 1].
        Shape: (N,), float32.
    class_names : List[str]
        Maps class ID (index) to class name string.
    mpp : float
        Microns per pixel at detection extraction level.
    """

    patch_id: str
    slide_x: int
    slide_y: int
    boxes_px: np.ndarray     # (N, 4) [x1, y1, x2, y2] patch coords, float32
    labels: np.ndarray       # (N,) int64
    scores: np.ndarray       # (N,) float32
    class_names: List[str]
    mpp: float

    def __len__(self) -> int:
        """Number of detected cells in this patch."""
        return len(self.labels)

    @property
    def centroids_px(self) -> np.ndarray:
        """Centroid coordinates (cx, cy) for each box in patch pixels.

        Returns
        -------
        np.ndarray
            Shape: (N, 2), float32. Columns: [cx, cy].
        """
        if len(self.boxes_px) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return (self.boxes_px[:, :2] + self.boxes_px[:, 2:]) / 2.0

    @property
    def centroids_slide_px(self) -> np.ndarray:
        """Centroid coordinates in slide-level (level 0) pixel space.

        Returns
        -------
        np.ndarray
            Shape: (N, 2), float32. Columns: [cx, cy] at level 0.
        """
        centroids = self.centroids_px.copy()
        centroids[:, 0] += self.slide_x
        centroids[:, 1] += self.slide_y
        return centroids

    def get_cells_by_class(self, class_id: int) -> "CellDetectionResult":
        """Return a new CellDetectionResult filtered to a single class.

        Parameters
        ----------
        class_id : int
            Cell class ID to filter to (see constants.CELL_CLASSES).

        Returns
        -------
        CellDetectionResult
            New result containing only cells with matching label.
        """
        mask = self.labels == class_id
        return CellDetectionResult(
            patch_id=self.patch_id,
            slide_x=self.slide_x,
            slide_y=self.slide_y,
            boxes_px=self.boxes_px[mask],
            labels=self.labels[mask],
            scores=self.scores[mask],
            class_names=self.class_names,
            mpp=self.mpp,
        )


# ---------------------------------------------------------------------------
# FUSION ENGINE OUTPUT TYPES
# ---------------------------------------------------------------------------

@dataclass
class PatchSTILScore:
    """sTIL score computed for a single patch.

    Attributes
    ----------
    patch_id : str
        Unique patch identifier.
    slide_x : int
        X coordinate of patch origin at level 0.
    slide_y : int
        Y coordinate of patch origin at level 0.
    n_lymphocytes_in_stroma : int
        Count of lymphocytes whose centroid falls in the stroma mask.
    stroma_area_um2 : float
        Total stroma area in this patch, in square micrometers.
    tumor_area_um2 : float
        Total tumor area in this patch, in square micrometers.
    stil_score : float
        sTIL density score for this patch (lymphocytes per mm² stroma).
        Note: Not directly comparable to clinical sTIL %; used for heatmap.
    is_scoreable : bool
        False if stroma area is below minimum threshold (score unreliable).
    """

    patch_id: str
    slide_x: int
    slide_y: int
    n_lymphocytes_in_stroma: int
    stroma_area_um2: float
    tumor_area_um2: float
    stil_score: float
    is_scoreable: bool


@dataclass
class sTILResult:
    """Final sTIL scoring result for a single Whole Slide Image.

    The primary output of the PathoAI pipeline. Contains the slide-level
    sTIL score, statistical confidence information, and full provenance.

    Attributes
    ----------
    slide_id : str
        Unique identifier of the scored slide.
    stil_score_pct : float
        Slide-level sTIL score in range [0, 100]. This is the primary output.
        Represents approximate percentage of stromal area occupied by lymphocytes.
    n_lymphocytes_in_stroma : int
        Total count of lymphocytes found in stromal regions across all patches.
    stroma_area_um2 : float
        Total stromal area across all scored patches, in μm².
    tumor_area_um2 : float
        Total tumor area across all scored patches, in μm².
    n_scoreable_patches : int
        Number of patches that had sufficient stroma for reliable scoring.
    patch_scores : np.ndarray
        Per-patch sTIL density scores (not percentage). Shape: (n_scoreable,).
        Used for spatial heatmap generation.
    patch_coordinates : List[Tuple[int, int]]
        Slide coordinates for each scoreable patch. Aligned with patch_scores.
    ci_lower : float
        Lower bound of 95% bootstrap confidence interval for stil_score_pct.
    ci_upper : float
        Upper bound of 95% bootstrap confidence interval for stil_score_pct.
    quality_flags : List[str]
        Quality flags assigned during validation. Empty list = clean result.
        Possible values: "LOW_CONFIDENCE", "INSUFFICIENT_STROMA",
        "INSUFFICIENT_LYMPHOCYTES", "SCORE_AT_BOUNDARY"
    processing_time_s : float
        Total pipeline processing time for this slide in seconds.
    seg_model_version : str
        Name and version of the segmentation model used.
    det_model_version : str
        Name and version of the detection model used.
    config_hash : str
        SHA-256 hash of the experiment configuration (for reproducibility).
    created_at : datetime
        Timestamp when the result was generated.
    """

    slide_id: str
    stil_score_pct: float
    n_lymphocytes_in_stroma: int
    stroma_area_um2: float
    tumor_area_um2: float
    n_scoreable_patches: int
    patch_scores: np.ndarray          # (n_scoreable,), float32
    patch_coordinates: List[Tuple[int, int]]  # aligned with patch_scores
    ci_lower: float
    ci_upper: float
    quality_flags: List[str] = field(default_factory=list)
    processing_time_s: float = 0.0
    seg_model_version: str = "unknown"
    det_model_version: str = "unknown"
    config_hash: str = "unknown"
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_confident(self) -> bool:
        """True if the result has no quality flags."""
        return len(self.quality_flags) == 0

    @property
    def ci_width(self) -> float:
        """Width of the 95% confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def stroma_area_mm2(self) -> float:
        """Total stroma area in mm²."""
        return self.stroma_area_um2 / 1e6

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dictionary (excludes numpy arrays)."""
        return {
            "schema_version": "1.0.0",
            "slide_id": self.slide_id,
            "stil_score_pct": round(self.stil_score_pct, 4),
            "n_lymphocytes_in_stroma": self.n_lymphocytes_in_stroma,
            "stroma_area_um2": round(self.stroma_area_um2, 2),
            "stroma_area_mm2": round(self.stroma_area_mm2, 6),
            "tumor_area_um2": round(self.tumor_area_um2, 2),
            "n_scoreable_patches": self.n_scoreable_patches,
            "confidence_interval_95": [
                round(self.ci_lower, 4),
                round(self.ci_upper, 4),
            ],
            "ci_width": round(self.ci_width, 4),
            "quality_flags": self.quality_flags,
            "is_confident": self.is_confident,
            "processing_time_s": round(self.processing_time_s, 2),
            "seg_model_version": self.seg_model_version,
            "det_model_version": self.det_model_version,
            "config_hash": self.config_hash,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# VALIDATION ENGINE OUTPUT TYPES
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Statistical validation report for a sTIL result.

    Attributes
    ----------
    slide_id : str
        Identifier of the validated slide.
    stil_score_pct : float
        The sTIL score being validated.
    ci_lower : float
        Bootstrap CI lower bound.
    ci_upper : float
        Bootstrap CI upper bound.
    bootstrap_n : int
        Number of bootstrap resamples used.
    passed : bool
        True if all quality gates passed.
    quality_flags : List[str]
        Descriptions of any quality issues found.
    pathologist_score : Optional[float]
        Ground truth pathologist score (if available for evaluation).
    absolute_error : Optional[float]
        |predicted - pathologist| (if pathologist_score is available).
    notes : str
        Human-readable summary of validation result.
    """

    slide_id: str
    stil_score_pct: float
    ci_lower: float
    ci_upper: float
    bootstrap_n: int
    passed: bool
    quality_flags: List[str] = field(default_factory=list)
    pathologist_score: Optional[float] = None
    absolute_error: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "slide_id": self.slide_id,
            "stil_score_pct": round(self.stil_score_pct, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "bootstrap_n": self.bootstrap_n,
            "passed": self.passed,
            "quality_flags": self.quality_flags,
            "pathologist_score": self.pathologist_score,
            "absolute_error": self.absolute_error,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# SHARED GEOMETRY AND PATHOLOGY DOMAIN MODELS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point:
    """A 2D coordinate point (x, y)."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box defined by row/col coordinates [min_y, min_x, max_y, max_x]."""
    min_y: int
    min_x: int
    max_y: int
    max_x: int

    @property
    def width(self) -> int:
        return self.max_x - self.min_x

    @property
    def height(self) -> int:
        return self.max_y - self.min_y

    def to_yxyx(self) -> List[int]:
        return [self.min_y, self.min_x, self.max_y, self.max_x]


@dataclass(frozen=True)
class Polygon:
    """Polygonal boundary represented by exterior (shell) and interior (holes) points lists."""
    exterior: List[Point]
    interiors: List[List[Point]] = field(default_factory=list)

    def to_list(self) -> List[List[Tuple[float, float]]]:
        """Convert polygon coordinates to list format."""
        coords = [[p.to_tuple() for p in self.exterior]]
        for hole in self.interiors:
            coords.append([p.to_tuple() for p in hole])
        return coords


@dataclass
class TumorROI:
    """A Region of Interest representing a classified tissue nest or connected region.

    Attributes
    ----------
    roi_id : int
        Unique identifier for the ROI.
    bbox : BoundingBox
        Bounding box bounds.
    centroid : Point
        Weighted centroid of the region.
    area_px : int
        Total area in pixels.
    area_um2 : float
        Physical area in square microns.
    perimeter_um : float
        Physical boundary perimeter in microns.
    contours : List[Polygon]
        Polygonal contours defining the region.
    eccentricity : float
        Measure of region elongation (0 = circle, 1 = line).
    solidity : float
        Ratio of pixels in region to pixels in convex hull.
    compactness : float
        Ratio of region area to perimeter squared (normalized relative to circle).
    equivalent_diameter_um : float
        Diameter of circle with equivalent area in microns.
    class_label : str
        Classification class name (e.g. 'tumor_bulk').
    """

    roi_id: int
    bbox: BoundingBox
    centroid: Point
    area_px: int
    area_um2: float
    perimeter_um: float
    contours: List[Polygon]
    eccentricity: float = 0.0
    solidity: float = 0.0
    compactness: float = 0.0
    equivalent_diameter_um: float = 0.0
    class_label: str = "tumor_bulk"


@dataclass
class CellDetection:
    """Single detected cell object with spatial and class metadata.

    Attributes
    ----------
    detection_id : str
        Unique identifier for this detection instance.
    slide_id : str
        Source slide identifier.
    roi_id : str
        Identifier of the parent TumorROI or region.
    bbox : BoundingBox
        Bounding box bounds in slide level-0 coordinates.
    centroid : Point
        Cell centroid coordinate in slide level-0 coordinates.
    confidence : float
        Detection confidence score [0, 1].
    class_id : int
        Integer class identifier.
    class_name : str
        Human-readable class name string (e.g. 'lymphocyte').
    area_pixels : float
        Area of detection bounding box in pixels.
    area_um2 : float
        Physical area of detection in square microns.
    """

    detection_id: str
    slide_id: str
    roi_id: str
    bbox: BoundingBox
    centroid: Point
    confidence: float
    class_id: int
    class_name: str
    area_pixels: float = 0.0
    area_um2: float = 0.0


@dataclass
class SpatialDetection:
    """Cell detection augmented with spatial relationship metadata relative to tissue ROIs.

    Attributes
    ----------
    detection : CellDetection
        Original typed cell detection instance.
    roi : TumorROI
        Associated parent TumorROI region.
    inside_tumor : bool
        True if cell centroid lies within tumor compartment.
    inside_stroma : bool
        True if cell centroid lies within tumor-associated stroma compartment.
    distance_to_tumor_boundary_um : float
        Distance in microns to nearest tumor region boundary.
    distance_to_roi_centroid_um : float
        Distance in microns to the ROI centroid.
    nearest_boundary_point : Point
        Nearest point on the ROI polygon boundary.
    spatial_label : str
        Classification label (e.g., 'intratumoral_lymphocyte', 'stromal_lymphocyte', 'distant_lymphocyte').
    metadata : Dict[str, Any]
        Additional key-value spatial provenance parameters.
    """

    detection: CellDetection
    roi: TumorROI
    inside_tumor: bool
    inside_stroma: bool
    distance_to_tumor_boundary_um: float
    distance_to_roi_centroid_um: float
    nearest_boundary_point: Point
    spatial_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpatialLabel(str, Enum):
    """Enumeration of spatial cell relationship categories."""
    INTRATUMORAL = "intratumoral"
    PERITUMORAL_STROMA = "peritumoral_stromal"
    DISTANT = "distant"
    OUTSIDE_TISSUE = "outside_tissue"


@dataclass
class FusionResult:
    """Encapsulates spatial fusion results and aggregate statistics.

    Attributes
    ----------
    slide_id : str
        Source slide identifier.
    spatial_detections : List[SpatialDetection]
        List of typed SpatialDetection objects.
    total_cells : int
        Total number of processed cell detections.
    intratumoral_cells : int
        Number of cells located inside tumor regions.
    stromal_cells : int
        Number of cells located inside tumor-associated stroma.
    distant_cells : int
        Number of cells located outside tumor and stroma.
    rejected_cells : int
        Number of detections rejected by distance or spatial filters.
    processing_time_s : float
        Total fusion execution time in seconds.
    metadata : Dict[str, Any]
        Additional key-value parameters.
    """

    slide_id: str
    spatial_detections: List[SpatialDetection]
    total_cells: int
    intratumoral_cells: int
    stromal_cells: int
    distant_cells: int
    rejected_cells: int = 0
    processing_time_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)




