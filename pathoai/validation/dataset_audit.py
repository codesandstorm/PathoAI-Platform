"""
pathoai/validation/dataset_audit.py
======================================
Statistical dataset audit for PathoAI-Platform.

Provides statistical profiling of the TIGER dataset beyond structural
validation: class distributions, pixel statistics, patient distributions,
empty-mask detection, and summary report generation.

This module is designed to run after `dataset_validator.validate_dataset()`
has confirmed structural integrity.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.7
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pathoai.core.constants import TISSUE_CLASSES
from pathoai.core.exceptions import DataError
from pathoai.core.logger import get_logger
from pathoai.core.utils.path_utils import list_files_with_extension, resolve_path

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MaskAuditResult:
    """Audit result for a single mask file.

    Attributes:
        stem: Slide stem name (filename without extension).
        path: Absolute path to the mask file.
        width: Mask width in pixels.
        height: Mask height in pixels.
        n_pixels: Total pixel count.
        class_pixel_counts: Dict mapping class_id (int) → pixel count.
        class_pixel_fractions: Dict mapping class_id (int) → fraction [0, 1].
        is_empty: True if no tissue class (class > 0) pixels exist.
        has_tumor: True if class 1 (tumor) pixels > 0.
        has_stroma: True if class 2 (stroma) pixels > 0.
        error: Error message if the mask could not be processed, else None.
    """
    stem: str
    path: Path
    width: int = 0
    height: int = 0
    n_pixels: int = 0
    class_pixel_counts: Dict[int, int] = field(default_factory=dict)
    class_pixel_fractions: Dict[int, float] = field(default_factory=dict)
    is_empty: bool = True
    has_tumor: bool = False
    has_stroma: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            "stem": self.stem,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "n_pixels": self.n_pixels,
            "class_pixel_counts": {str(k): v for k, v in self.class_pixel_counts.items()},
            "class_pixel_fractions": {
                str(k): round(v, 6) for k, v in self.class_pixel_fractions.items()
            },
            "is_empty": self.is_empty,
            "has_tumor": self.has_tumor,
            "has_stroma": self.has_stroma,
            "error": self.error,
        }


@dataclass
class DatasetAuditReport:
    """Complete statistical audit report for the dataset.

    Attributes:
        dataset_root: Absolute path to the dataset root.
        n_masks_audited: Number of masks successfully audited.
        n_masks_failed: Number of masks that could not be processed.
        n_empty_masks: Number of masks with no tissue pixels.
        n_no_stroma: Masks with tumor but no stroma (cannot compute sTIL).
        overall_class_distribution: Aggregate pixel counts per class across all masks.
        overall_class_fractions: Aggregate pixel fractions per class.
        mask_results: Per-mask audit results.
    """
    dataset_root: Path
    n_masks_audited: int = 0
    n_masks_failed: int = 0
    n_empty_masks: int = 0
    n_no_stroma: int = 0
    overall_class_distribution: Dict[int, int] = field(default_factory=dict)
    overall_class_fractions: Dict[int, float] = field(default_factory=dict)
    mask_results: List[MaskAuditResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            "dataset_root": str(self.dataset_root),
            "n_masks_audited": self.n_masks_audited,
            "n_masks_failed": self.n_masks_failed,
            "n_empty_masks": self.n_empty_masks,
            "n_no_stroma": self.n_no_stroma,
            "overall_class_distribution": {
                str(k): {"count": v, "class_name": TISSUE_CLASSES.get(k, "unknown")}
                for k, v in self.overall_class_distribution.items()
            },
            "overall_class_fractions": {
                str(k): round(v, 6) for k, v in self.overall_class_fractions.items()
            },
            "mask_results": [r.to_dict() for r in self.mask_results],
        }

    def save(self, path: Path) -> None:
        """Save the audit report to a JSON file.

        Args:
            path: Output path for the JSON file. Parent is created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Audit report saved", extra={"path": str(path)})


# ---------------------------------------------------------------------------
# Individual audit functions
# ---------------------------------------------------------------------------

def audit_mask(mask_path: Path) -> MaskAuditResult:
    """Compute pixel-level class statistics for a single mask file.

    Args:
        mask_path: Path to the mask PNG file. Expected to contain integer
            class IDs per pixel (single-channel or mode-converted).

    Returns:
        MaskAuditResult with pixel counts and fractions per class.
    """
    result = MaskAuditResult(
        stem=mask_path.stem,
        path=mask_path,
    )

    try:
        from PIL import Image
        with Image.open(mask_path) as img:
            # Convert to grayscale to get class IDs
            if img.mode != "L":
                img = img.convert("L")
            mask_array = np.array(img, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        logger.warning("Cannot open mask: %s — %s", mask_path, exc)
        return result

    result.height, result.width = mask_array.shape
    result.n_pixels = int(mask_array.size)

    # Count pixels per class
    unique, counts = np.unique(mask_array, return_counts=True)
    for class_id, count in zip(unique.tolist(), counts.tolist()):
        result.class_pixel_counts[class_id] = int(count)
        result.class_pixel_fractions[class_id] = (
            count / result.n_pixels if result.n_pixels > 0 else 0.0
        )

    # Derived flags
    non_background = result.n_pixels - result.class_pixel_counts.get(0, 0)
    result.is_empty = non_background == 0
    result.has_tumor = result.class_pixel_counts.get(1, 0) > 0
    result.has_stroma = result.class_pixel_counts.get(2, 0) > 0

    logger.debug(
        "Mask audited",
        extra={
            "stem": result.stem,
            "is_empty": result.is_empty,
            "has_stroma": result.has_stroma,
        },
    )
    return result


def compute_class_distribution(
    mask_results: List[MaskAuditResult],
    n_classes: int = 6,
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Aggregate per-mask class distributions into dataset-level statistics.

    Args:
        mask_results: List of per-mask audit results.
        n_classes: Number of tissue classes. Defaults to 6 (TIGER schema).

    Returns:
        Tuple of (overall_counts dict, overall_fractions dict).
        Both keyed by class ID (int).
    """
    totals: Dict[int, int] = {c: 0 for c in range(n_classes)}

    for result in mask_results:
        if result.error is not None:
            continue
        for class_id, count in result.class_pixel_counts.items():
            if class_id in totals:
                totals[class_id] += count
            else:
                totals[class_id] = count

    grand_total = sum(totals.values())
    fractions: Dict[int, float] = {
        c: (totals[c] / grand_total if grand_total > 0 else 0.0)
        for c in totals
    }

    return totals, fractions


def detect_empty_masks(mask_results: List[MaskAuditResult]) -> List[str]:
    """Return stem names of masks with no tissue pixels.

    Args:
        mask_results: List of per-mask audit results.

    Returns:
        List of stem names for empty masks.
    """
    return [r.stem for r in mask_results if r.is_empty and r.error is None]


def detect_scorability_issues(mask_results: List[MaskAuditResult]) -> List[str]:
    """Return stems of masks that have tumor but no stroma.

    Slides with tumor but no annotated stroma cannot produce a valid sTIL
    score under the Salgado et al. (2015) methodology.

    Args:
        mask_results: List of per-mask audit results.

    Returns:
        List of stem names for slides with scorability issues.
    """
    return [
        r.stem
        for r in mask_results
        if r.error is None and r.has_tumor and not r.has_stroma
    ]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def audit_dataset(
    dataset_root: str | Path,
    mask_subdir: str = "masks",
    mask_extension: str = ".png",
    n_classes: int = 6,
    save_report_to: Optional[Path] = None,
) -> DatasetAuditReport:
    """Run a full statistical audit of the dataset masks.

    Iterates over all mask files, computes per-mask class statistics,
    aggregates dataset-level distributions, and identifies problematic
    slides (empty masks, scorability issues).

    Args:
        dataset_root: Root directory of the dataset.
        mask_subdir: Subdirectory containing mask files. Defaults to
            ``"masks"``.
        mask_extension: File extension for masks. Defaults to ``".png"``.
        n_classes: Number of tissue classes. Defaults to 6.
        save_report_to: If provided, save JSON report to this path.

    Returns:
        DatasetAuditReport with complete statistical profile.

    Raises:
        DataError: If the masks directory does not exist.

    Example:
        >>> report = audit_dataset("data/raw/tiger")
        >>> print(f"Empty masks: {report.n_empty_masks}")
        >>> print(f"Class distribution: {report.overall_class_fractions}")
    """
    root = resolve_path(dataset_root)
    masks_dir = root / mask_subdir

    if not masks_dir.is_dir():
        raise DataError(
            f"Masks directory not found: {masks_dir}. "
            f"Run validate_dataset() first to confirm directory structure."
        )

    report = DatasetAuditReport(dataset_root=root)

    mask_paths = list_files_with_extension(masks_dir, mask_extension, recursive=True)

    if not mask_paths:
        logger.warning("No mask files found in %s", masks_dir)
        return report

    logger.info(
        "Starting dataset audit",
        extra={"n_masks": len(mask_paths), "dataset_root": str(root)},
    )

    # Audit each mask
    for mask_path in mask_paths:
        result = audit_mask(mask_path)
        report.mask_results.append(result)
        if result.error is not None:
            report.n_masks_failed += 1
        else:
            report.n_masks_audited += 1

    # Aggregate statistics
    totals, fractions = compute_class_distribution(report.mask_results, n_classes)
    report.overall_class_distribution = totals
    report.overall_class_fractions = fractions

    # Identify problem slides
    empty_stems = detect_empty_masks(report.mask_results)
    report.n_empty_masks = len(empty_stems)
    if empty_stems:
        logger.warning(
            "Empty masks detected",
            extra={"n_empty": len(empty_stems), "stems": empty_stems[:5]},
        )

    scorability_issues = detect_scorability_issues(report.mask_results)
    report.n_no_stroma = len(scorability_issues)
    if scorability_issues:
        logger.warning(
            "Slides with tumor but no stroma detected",
            extra={"n_slides": len(scorability_issues)},
        )

    logger.info(
        "Dataset audit complete",
        extra={
            "n_audited": report.n_masks_audited,
            "n_failed": report.n_masks_failed,
            "n_empty": report.n_empty_masks,
            "n_no_stroma": report.n_no_stroma,
        },
    )

    if save_report_to:
        report.save(save_report_to)

    return report
