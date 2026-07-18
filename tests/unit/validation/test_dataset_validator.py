"""
tests/unit/validation/test_dataset_validator.py
==================================================
Unit tests for pathoai.validation.dataset_validator.

Tests cover all data types and validation functions using only
synthetic in-memory files — no real WSI files required.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.6
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from pathoai.validation.dataset_validator import (
    DatasetValidationReport,
    IssueSeverity,
    ValidationIssue,
    detect_duplicate_slides,
    validate_dataset,
    validate_folder_structure,
    validate_image_integrity,
    validate_masks,
    validate_slides,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slide(directory: Path, name: str, content: bytes = b"SVSDUMMY") -> Path:
    """Write a minimal fake WSI file to a directory."""
    p = directory / name
    p.write_bytes(content)
    return p


def _make_mask(directory: Path, name: str) -> Path:
    """Write a valid 64×64 grayscale PNG mask."""
    import numpy as np
    mask_data = (np.ones((64, 64), dtype=np.uint8) * 2)  # All-stroma
    img = Image.fromarray(mask_data, mode="L")
    p = directory / name
    img.save(p)
    return p


def _make_dataset(root: Path, n_slides: int = 2) -> tuple[list[Path], list[Path]]:
    """Create a valid synthetic TIGER-style dataset structure."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    slides, masks = [], []
    for i in range(n_slides):
        slide = _make_slide(root / "images", f"slide_{i:03d}.tif")
        mask = _make_mask(root / "masks", f"slide_{i:03d}.png")
        slides.append(slide)
        masks.append(mask)
    return slides, masks


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------

class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_construction(self):
        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            category="missing_directory",
            message="required dir missing",
        )
        assert issue.severity == IssueSeverity.ERROR
        assert issue.category == "missing_directory"

    def test_to_dict_contains_required_keys(self):
        issue = ValidationIssue(
            severity=IssueSeverity.WARNING,
            category="test",
            message="test message",
            path=Path("/some/path.png"),
        )
        d = issue.to_dict()
        for key in ("severity", "category", "message", "path"):
            assert key in d

    def test_path_is_none_when_not_provided(self):
        issue = ValidationIssue(severity=IssueSeverity.INFO, category="x", message="y")
        assert issue.to_dict()["path"] is None


# ---------------------------------------------------------------------------
# DatasetValidationReport
# ---------------------------------------------------------------------------

class TestDatasetValidationReport:
    """Tests for DatasetValidationReport dataclass."""

    def test_errors_property_filters_error_severity(self, tmp_path: Path):
        report = DatasetValidationReport(dataset_root=tmp_path)
        report.issues.append(ValidationIssue(IssueSeverity.ERROR, "cat", "msg"))
        report.issues.append(ValidationIssue(IssueSeverity.WARNING, "cat", "msg"))
        assert len(report.errors) == 1
        assert len(report.warnings) == 1

    def test_to_dict_is_json_serializable(self, tmp_path: Path):
        report = DatasetValidationReport(dataset_root=tmp_path)
        json_str = json.dumps(report.to_dict(), default=str)
        assert json.loads(json_str) is not None

    def test_save_creates_json_file(self, tmp_path: Path):
        report = DatasetValidationReport(dataset_root=tmp_path)
        out = tmp_path / "report.json"
        report.save(out)
        assert out.exists()
        with open(out) as f:
            loaded = json.load(f)
        assert "dataset_root" in loaded


# ---------------------------------------------------------------------------
# validate_folder_structure
# ---------------------------------------------------------------------------

class TestValidateFolderStructure:
    """Tests for validate_folder_structure()."""

    def test_no_issues_when_all_dirs_present(self, tmp_path: Path):
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        issues = validate_folder_structure(tmp_path)
        errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
        assert errors == []

    def test_error_when_images_missing(self, tmp_path: Path):
        (tmp_path / "masks").mkdir()
        issues = validate_folder_structure(tmp_path)
        errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
        assert any("images" in i.message for i in errors)

    def test_error_when_masks_missing(self, tmp_path: Path):
        (tmp_path / "images").mkdir()
        issues = validate_folder_structure(tmp_path)
        errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
        assert any("masks" in i.message for i in errors)

    def test_info_when_optional_dir_absent(self, tmp_path: Path):
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        issues = validate_folder_structure(tmp_path)
        info_issues = [i for i in issues if i.severity == IssueSeverity.INFO]
        assert len(info_issues) >= 1  # annotations/ is optional


# ---------------------------------------------------------------------------
# validate_slides
# ---------------------------------------------------------------------------

class TestValidateSlides:
    """Tests for validate_slides()."""

    def test_discovers_tif_slides(self, tmp_path: Path):
        slides_dir = tmp_path / "images"
        slides_dir.mkdir()
        _make_slide(slides_dir, "A001.tif")
        _make_slide(slides_dir, "A002.tif")
        issues, paths = validate_slides(slides_dir, extensions=[".tif"])
        assert issues == []
        assert len(paths) == 2

    def test_error_when_no_slides_found(self, tmp_path: Path):
        slides_dir = tmp_path / "images"
        slides_dir.mkdir()
        issues, paths = validate_slides(slides_dir, extensions=[".tif"])
        assert any(i.severity == IssueSeverity.ERROR for i in issues)
        assert paths == []

    def test_error_when_directory_missing(self, tmp_path: Path):
        issues, paths = validate_slides(tmp_path / "nonexistent")
        assert any(i.severity == IssueSeverity.ERROR for i in issues)
        assert paths == []

    def test_returns_sorted_paths(self, tmp_path: Path):
        slides_dir = tmp_path / "images"
        slides_dir.mkdir()
        _make_slide(slides_dir, "Z.tif")
        _make_slide(slides_dir, "A.tif")
        _, paths = validate_slides(slides_dir, extensions=[".tif"])
        assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# validate_masks
# ---------------------------------------------------------------------------

class TestValidateMasks:
    """Tests for validate_masks()."""

    def test_no_issues_when_all_masks_present(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        _make_mask(masks_dir, "slide_001.png")
        _make_mask(masks_dir, "slide_002.png")
        issues, n_missing = validate_masks(masks_dir, ["slide_001", "slide_002"])
        assert issues == []
        assert n_missing == 0

    def test_error_for_missing_mask(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        _make_mask(masks_dir, "slide_001.png")
        issues, n_missing = validate_masks(masks_dir, ["slide_001", "slide_002"])
        assert n_missing == 1
        assert any(i.severity == IssueSeverity.ERROR for i in issues)

    def test_error_when_masks_dir_missing(self, tmp_path: Path):
        issues, n_missing = validate_masks(
            tmp_path / "nonexistent", ["slide_001"]
        )
        assert any(i.severity == IssueSeverity.ERROR for i in issues)

    def test_n_missing_matches_error_count(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        issues, n_missing = validate_masks(masks_dir, ["A", "B", "C"])
        assert n_missing == 3
        error_count = sum(
            1 for i in issues
            if i.severity == IssueSeverity.ERROR and "No mask" in i.message
        )
        assert error_count == 3


# ---------------------------------------------------------------------------
# validate_image_integrity
# ---------------------------------------------------------------------------

class TestValidateImageIntegrity:
    """Tests for validate_image_integrity()."""

    def test_no_issues_for_valid_png_masks(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        p1 = _make_mask(masks_dir, "a.png")
        p2 = _make_mask(masks_dir, "b.png")
        issues = validate_image_integrity([p1, p2], check_masks=True)
        assert issues == []

    def test_empty_list_when_check_masks_is_false(self, tmp_path: Path):
        """When check_masks=False, all paths are skipped."""
        p = tmp_path / "slide.tif"
        p.write_bytes(b"dummy")
        issues = validate_image_integrity([p], check_masks=False)
        assert issues == []

    def test_error_for_corrupt_image(self, tmp_path: Path):
        """A zero-byte file should be flagged as corrupt."""
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"NOT_A_PNG_AT_ALL")
        issues = validate_image_integrity([corrupt], check_masks=True)
        assert any(i.severity == IssueSeverity.ERROR for i in issues)


# ---------------------------------------------------------------------------
# detect_duplicate_slides
# ---------------------------------------------------------------------------

class TestDetectDuplicateSlides:
    """Tests for detect_duplicate_slides()."""

    def test_no_duplicates_for_unique_files(self, tmp_path: Path):
        slides_dir = tmp_path / "images"
        slides_dir.mkdir()
        s1 = _make_slide(slides_dir, "s1.tif", content=b"CONTENT_UNIQUE_1")
        s2 = _make_slide(slides_dir, "s2.tif", content=b"CONTENT_UNIQUE_2")
        issues = detect_duplicate_slides([s1, s2])
        assert issues == []

    def test_warning_for_duplicate_files(self, tmp_path: Path):
        slides_dir = tmp_path / "images"
        slides_dir.mkdir()
        content = b"A" * 200  # Same content
        s1 = _make_slide(slides_dir, "s1.tif", content=content)
        s2 = _make_slide(slides_dir, "s2.tif", content=content)
        issues = detect_duplicate_slides([s1, s2])
        assert any(i.severity == IssueSeverity.WARNING for i in issues)

    def test_empty_list_for_single_slide(self, tmp_path: Path):
        slides_dir = tmp_path / "images"
        slides_dir.mkdir()
        s = _make_slide(slides_dir, "s.tif")
        issues = detect_duplicate_slides([s])
        assert issues == []


# ---------------------------------------------------------------------------
# validate_dataset (orchestrator)
# ---------------------------------------------------------------------------

class TestValidateDataset:
    """Tests for the validate_dataset() orchestrator."""

    def test_valid_dataset_returns_is_valid_true(self, tmp_path: Path):
        _make_dataset(tmp_path, n_slides=3)
        report = validate_dataset(
            tmp_path,
            check_image_integrity=True,
            check_duplicates=True,
        )
        assert report.is_valid is True

    def test_returns_correct_slide_count(self, tmp_path: Path):
        _make_dataset(tmp_path, n_slides=4)
        report = validate_dataset(tmp_path, check_image_integrity=False)
        assert report.n_slides_found == 4

    def test_missing_masks_dir_causes_error(self, tmp_path: Path):
        (tmp_path / "images").mkdir()
        _make_slide(tmp_path / "images", "slide_001.tif")
        report = validate_dataset(tmp_path, check_image_integrity=False)
        assert report.is_valid is False
        assert len(report.errors) > 0

    def test_missing_slide_mask_causes_error(self, tmp_path: Path):
        _make_dataset(tmp_path, n_slides=2)
        # Add a slide without a corresponding mask
        _make_slide(tmp_path / "images", "orphan.tif")
        report = validate_dataset(tmp_path, check_image_integrity=False)
        assert report.is_valid is False

    def test_saves_report_when_path_provided(self, tmp_path: Path):
        _make_dataset(tmp_path)
        out = tmp_path / "report.json"
        validate_dataset(tmp_path, save_report_to=out, check_image_integrity=False)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert "is_valid" in loaded

    def test_report_contains_correct_mask_count(self, tmp_path: Path):
        _make_dataset(tmp_path, n_slides=5)
        report = validate_dataset(tmp_path, check_image_integrity=False)
        assert report.n_masks_found == 5
