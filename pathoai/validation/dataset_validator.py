"""
pathoai/validation/dataset_validator.py
========================================
Dataset integrity validation for PathoAI-Platform.

Validates the TIGER dataset directory structure, image files,
mask files, and annotation CSV before any training or inference begins.

Design principles:
    - Collect all errors before reporting (do not fail-fast on first issue).
    - Every check produces a structured ValidationIssue with severity.
    - Callers decide whether to abort on WARNING vs. ERROR level issues.
    - All filesystem operations go through path_utils for consistent logging.

Reference:
    TIGER Grand Challenge data format:
    https://tiger.grand-challenge.org/Data/

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.6
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from pathoai.core.constants import SUPPORTED_WSI_FORMATS
from pathoai.core.exceptions import DatasetValidationError
from pathoai.core.logger import get_logger
from pathoai.core.utils.path_utils import (
    ensure_directory,
    list_files_with_extension,
    resolve_path,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Issue severity and result types
# ---------------------------------------------------------------------------

class IssueSeverity(str, Enum):
    """Severity level for a dataset validation issue."""
    ERROR = "ERROR"       # Blocks training/inference — must be resolved
    WARNING = "WARNING"   # May affect quality — should be resolved
    INFO = "INFO"         # Informational only


@dataclass
class ValidationIssue:
    """A single dataset validation issue.

    Attributes:
        severity: ERROR, WARNING, or INFO.
        category: Short string identifying the check that raised the issue
            (e.g., ``"missing_slide"``, ``"corrupt_mask"``).
        message: Human-readable description of the issue.
        path: Optional path to the file that caused the issue.
    """
    severity: IssueSeverity
    category: str
    message: str
    path: Optional[Path] = None

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "path": str(self.path) if self.path else None,
        }


@dataclass
class DatasetValidationReport:
    """Complete dataset validation report.

    Attributes:
        dataset_root: Absolute path to the validated dataset root.
        issues: All issues found during validation, sorted by severity.
        n_slides_found: Total number of WSI files found.
        n_masks_found: Total number of mask files found.
        n_slides_without_mask: Slides that have no corresponding mask.
        n_duplicate_slides: Slides with duplicate SHA-256 hashes.
        is_valid: True if there are no ERROR-severity issues.
    """
    dataset_root: Path
    issues: List[ValidationIssue] = field(default_factory=list)
    n_slides_found: int = 0
    n_masks_found: int = 0
    n_slides_without_mask: int = 0
    n_duplicate_slides: int = 0
    is_valid: bool = False

    @property
    def errors(self) -> List[ValidationIssue]:
        """Return only ERROR-severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Return only WARNING-severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            "dataset_root": str(self.dataset_root),
            "is_valid": self.is_valid,
            "n_slides_found": self.n_slides_found,
            "n_masks_found": self.n_masks_found,
            "n_slides_without_mask": self.n_slides_without_mask,
            "n_duplicate_slides": self.n_duplicate_slides,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }

    def save(self, path: Path) -> None:
        """Save the report to a JSON file.

        Args:
            path: Output file path. Parent directory is created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Validation report saved", extra={"path": str(path)})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file using buffered reads.

    Args:
        path: Path to the file to hash.
        chunk_size: Read chunk size in bytes. Defaults to 64 KiB.

    Returns:
        Lowercase hex digest of the SHA-256 hash.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _try_open_image(path: Path) -> Optional[str]:
    """Try to open an image file; return an error string if it fails.

    Uses PIL/Pillow for non-WSI images (masks, thumbnails).

    Args:
        path: Path to the image file.

    Returns:
        None if the file opens successfully; an error message string if it fails.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


# ---------------------------------------------------------------------------
# Individual validation checks
# ---------------------------------------------------------------------------

def validate_folder_structure(dataset_root: Path) -> List[ValidationIssue]:
    """Validate that required TIGER dataset subdirectories exist.

    The expected structure is::

        dataset_root/
        ├── images/          # WSI files (.tif, .svs, etc.)
        ├── masks/           # Tissue mask annotations (PNG)
        └── annotations/     # Optional: CSV with sTIL scores

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        List of ValidationIssues for any missing directories.
    """
    issues: List[ValidationIssue] = []
    required_dirs = ["images", "masks"]
    optional_dirs = ["annotations"]

    for name in required_dirs:
        target = dataset_root / name
        if not target.is_dir():
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category="missing_directory",
                message=f"Required directory missing: {target}",
                path=target,
            ))
        else:
            logger.debug("Directory present: %s", target)

    for name in optional_dirs:
        target = dataset_root / name
        if not target.is_dir():
            issues.append(ValidationIssue(
                severity=IssueSeverity.INFO,
                category="optional_directory_absent",
                message=f"Optional directory not found (expected if using CSV splits): {target}",
                path=target,
            ))

    return issues


def validate_slides(
    slides_dir: Path,
    extensions: Optional[Sequence[str]] = None,
) -> tuple[List[ValidationIssue], List[Path]]:
    """Validate WSI files in the slides directory.

    Checks that:
    - At least one slide exists.
    - All slides have a supported extension.

    Args:
        slides_dir: Directory containing WSI files.
        extensions: File extensions to accept. Defaults to
            ``SUPPORTED_WSI_FORMATS``.

    Returns:
        Tuple of (issues list, list of all valid slide paths found).
    """
    issues: List[ValidationIssue] = []
    if extensions is None:
        extensions = list(SUPPORTED_WSI_FORMATS)

    if not slides_dir.is_dir():
        issues.append(ValidationIssue(
            severity=IssueSeverity.ERROR,
            category="missing_slides_directory",
            message=f"Slides directory not found: {slides_dir}",
            path=slides_dir,
        ))
        return issues, []

    slide_paths: List[Path] = []
    for ext in extensions:
        slide_paths.extend(list_files_with_extension(slides_dir, ext, recursive=True))
    slide_paths = sorted(set(slide_paths))

    if not slide_paths:
        issues.append(ValidationIssue(
            severity=IssueSeverity.ERROR,
            category="no_slides_found",
            message=(
                f"No WSI files found in {slides_dir}. "
                f"Supported extensions: {sorted(extensions)}"
            ),
            path=slides_dir,
        ))
    else:
        logger.info(
            "Slides discovered",
            extra={"n_slides": len(slide_paths), "directory": str(slides_dir)},
        )

    return issues, slide_paths


def validate_masks(
    masks_dir: Path,
    slide_stems: Sequence[str],
    mask_extension: str = ".png",
) -> tuple[List[ValidationIssue], int]:
    """Validate that every slide has a corresponding mask file.

    Args:
        masks_dir: Directory containing mask files.
        slide_stems: Stem names (without extension) of all discovered slides.
        mask_extension: Extension of mask files. Defaults to ``".png"``.

    Returns:
        Tuple of (issues list, count of slides missing a mask).
    """
    issues: List[ValidationIssue] = []
    n_missing = 0

    if not masks_dir.is_dir():
        issues.append(ValidationIssue(
            severity=IssueSeverity.ERROR,
            category="missing_masks_directory",
            message=f"Masks directory not found: {masks_dir}",
            path=masks_dir,
        ))
        return issues, len(slide_stems)

    mask_stems: Set[str] = {
        p.stem
        for p in list_files_with_extension(masks_dir, mask_extension, recursive=True)
    }

    for stem in slide_stems:
        if stem not in mask_stems:
            n_missing += 1
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category="missing_mask",
                message=f"No mask found for slide '{stem}' in {masks_dir}",
                path=masks_dir / f"{stem}{mask_extension}",
            ))

    if n_missing == 0:
        logger.info(
            "All masks present",
            extra={"n_slides": len(slide_stems), "directory": str(masks_dir)},
        )
    else:
        logger.warning(
            "Missing masks detected",
            extra={"n_missing": n_missing, "n_total": len(slide_stems)},
        )

    return issues, n_missing


def validate_image_integrity(
    image_paths: Sequence[Path],
    *,
    check_masks: bool = False,
) -> List[ValidationIssue]:
    """Validate image files for corruption using PIL/Pillow.

    Skips WSI files (which cannot be opened by PIL) and checks only
    mask/thumbnail PNG/JPEG files when ``check_masks=True``.

    Args:
        image_paths: Paths to image files to validate.
        check_masks: If True, validate files as raster images (PNG/JPEG)
            using PIL. If False (default), skips PIL verification
            (used when paths are WSI files that require OpenSlide).

    Returns:
        List of ValidationIssues for any unreadable files.
    """
    issues: List[ValidationIssue] = []

    if not check_masks:
        logger.debug(
            "Skipping PIL integrity check for WSI files — use OpenSlide for WSI validation"
        )
        return issues

    for path in image_paths:
        error = _try_open_image(path)
        if error is not None:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category="corrupt_image",
                message=f"Cannot open image (possibly corrupt): {path}. Error: {error}",
                path=path,
            ))

    if not issues:
        logger.info(
            "Image integrity check passed",
            extra={"n_images": len(image_paths)},
        )

    return issues


def detect_duplicate_slides(slide_paths: Sequence[Path]) -> List[ValidationIssue]:
    """Detect duplicate WSI files by comparing SHA-256 hashes.

    Note: Hashing large WSI files is slow. This function hashes only
    the first 1 MiB of each file for a fast approximate check.

    Args:
        slide_paths: Paths to WSI files to compare.

    Returns:
        List of ValidationIssues for any detected duplicates.

    Raises:
        DatasetValidationError: If a file cannot be read during hashing.
    """
    issues: List[ValidationIssue] = []
    hash_to_paths: Dict[str, List[Path]] = {}
    SAMPLE_BYTES = 1024 * 1024  # 1 MiB sample

    for path in slide_paths:
        try:
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                sha.update(f.read(SAMPLE_BYTES))
            digest = sha.hexdigest()
        except OSError as exc:
            raise DatasetValidationError(
                f"Cannot read slide file for duplicate detection: {path}"
            ) from exc

        if digest not in hash_to_paths:
            hash_to_paths[digest] = []
        hash_to_paths[digest].append(path)

    for digest, paths in hash_to_paths.items():
        if len(paths) > 1:
            path_strs = ", ".join(str(p) for p in paths)
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category="duplicate_slide",
                message=(
                    f"Possible duplicate slides detected (same first-MiB hash: {digest[:8]}...): "
                    f"{path_strs}"
                ),
                path=paths[0],
            ))

    if issues:
        logger.warning("Duplicate slides detected", extra={"n_groups": len(issues)})
    else:
        logger.debug("No duplicate slides detected (%d slides checked)", len(slide_paths))

    return issues


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def validate_dataset(
    dataset_root: str | Path,
    *,
    slide_subdir: str = "images",
    mask_subdir: str = "masks",
    mask_extension: str = ".png",
    check_image_integrity: bool = True,
    check_duplicates: bool = True,
    extensions: Optional[Sequence[str]] = None,
    save_report_to: Optional[Path] = None,
) -> DatasetValidationReport:
    """Run all dataset validation checks and return a structured report.

    This is the primary entry point for dataset validation. It orchestrates
    all individual checks and collects results into a single report.

    Args:
        dataset_root: Root directory of the TIGER dataset.
        slide_subdir: Subdirectory name containing WSI files. Defaults
            to ``"images"``.
        mask_subdir: Subdirectory name containing mask files. Defaults
            to ``"masks"``.
        mask_extension: File extension for mask files. Defaults to
            ``".png"``.
        check_image_integrity: If True, validate mask files using PIL.
            Defaults to True.
        check_duplicates: If True, run duplicate slide detection.
            Defaults to True.
        extensions: WSI file extensions to search for. Defaults to
            ``SUPPORTED_WSI_FORMATS``.
        save_report_to: If provided, save the JSON report to this path.

    Returns:
        DatasetValidationReport with all issues and summary statistics.

    Example:
        >>> report = validate_dataset("data/raw/tiger")
        >>> if not report.is_valid:
        ...     for issue in report.errors:
        ...         print(issue.message)
    """
    root = resolve_path(dataset_root)
    report = DatasetValidationReport(dataset_root=root)

    logger.info("Starting dataset validation", extra={"dataset_root": str(root)})

    # 1. Folder structure
    folder_issues = validate_folder_structure(root)
    report.issues.extend(folder_issues)

    # If root directories are missing, many subsequent checks will fail —
    # return early with the folder-structure errors.
    slides_dir = root / slide_subdir
    masks_dir = root / mask_subdir
    if not slides_dir.is_dir() or not masks_dir.is_dir():
        report.is_valid = len(report.errors) == 0
        if save_report_to:
            report.save(save_report_to)
        return report

    # 2. Slide discovery
    slide_issues, slide_paths = validate_slides(slides_dir, extensions=extensions)
    report.issues.extend(slide_issues)
    report.n_slides_found = len(slide_paths)

    if not slide_paths:
        report.is_valid = len(report.errors) == 0
        if save_report_to:
            report.save(save_report_to)
        return report

    # 3. Mask validation
    slide_stems = [p.stem for p in slide_paths]
    mask_issues, n_missing = validate_masks(masks_dir, slide_stems, mask_extension)
    report.issues.extend(mask_issues)
    report.n_slides_without_mask = n_missing

    mask_paths = list_files_with_extension(masks_dir, mask_extension, recursive=True)
    report.n_masks_found = len(mask_paths)

    # 4. Image integrity (masks only — WSI requires OpenSlide)
    if check_image_integrity and mask_paths:
        integrity_issues = validate_image_integrity(
            mask_paths, check_masks=True
        )
        report.issues.extend(integrity_issues)

    # 5. Duplicate detection
    if check_duplicates and len(slide_paths) > 1:
        dup_issues = detect_duplicate_slides(slide_paths)
        report.issues.extend(dup_issues)
        report.n_duplicate_slides = len(dup_issues)

    # 6. Finalize
    report.is_valid = len(report.errors) == 0

    logger.info(
        "Dataset validation complete",
        extra={
            "is_valid": report.is_valid,
            "n_slides": report.n_slides_found,
            "n_masks": report.n_masks_found,
            "n_errors": len(report.errors),
            "n_warnings": len(report.warnings),
        },
    )

    if save_report_to:
        report.save(save_report_to)

    return report
