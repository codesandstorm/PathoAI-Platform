"""
tests/unit/validation/test_dataset_audit.py
=============================================
Unit tests for pathoai.validation.dataset_audit.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.7
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pathoai.validation.dataset_audit import (
    DatasetAuditReport,
    MaskAuditResult,
    audit_dataset,
    audit_mask,
    compute_class_distribution,
    detect_empty_masks,
    detect_scorability_issues,
)
from pathoai.core.exceptions import DataError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(directory: Path, name: str, class_ids: np.ndarray | None = None) -> Path:
    """Write a PNG mask with given class_id pixel values."""
    if class_ids is None:
        class_ids = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    img = Image.fromarray(class_ids, mode="L")
    p = directory / name
    img.save(p)
    return p


def _make_empty_mask(directory: Path, name: str) -> Path:
    """Write a mask where all pixels are background (class 0)."""
    data = np.zeros((16, 16), dtype=np.uint8)
    img = Image.fromarray(data, mode="L")
    p = directory / name
    img.save(p)
    return p


def _make_tumor_only_mask(directory: Path, name: str) -> Path:
    """Write a mask with only tumor pixels (class 1) — no stroma."""
    data = np.ones((16, 16), dtype=np.uint8)
    img = Image.fromarray(data, mode="L")
    p = directory / name
    img.save(p)
    return p


def _make_tumor_stroma_mask(directory: Path, name: str) -> Path:
    """Write a mask with both tumor (1) and stroma (2)."""
    data = np.zeros((16, 16), dtype=np.uint8)
    data[:8, :] = 1   # Tumor
    data[8:, :] = 2   # Stroma
    img = Image.fromarray(data, mode="L")
    p = directory / name
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# MaskAuditResult
# ---------------------------------------------------------------------------

class TestMaskAuditResult:
    """Tests for MaskAuditResult dataclass."""

    def test_to_dict_is_json_serializable(self, tmp_path: Path):
        result = MaskAuditResult(stem="test", path=tmp_path / "test.png")
        json_str = json.dumps(result.to_dict(), default=str)
        assert json.loads(json_str) is not None

    def test_class_ids_are_string_keys_in_to_dict(self, tmp_path: Path):
        result = MaskAuditResult(
            stem="test",
            path=tmp_path / "test.png",
            class_pixel_counts={0: 100, 1: 50},
        )
        d = result.to_dict()
        assert "0" in d["class_pixel_counts"]
        assert "1" in d["class_pixel_counts"]


# ---------------------------------------------------------------------------
# DatasetAuditReport
# ---------------------------------------------------------------------------

class TestDatasetAuditReport:
    """Tests for DatasetAuditReport dataclass."""

    def test_to_dict_contains_required_keys(self, tmp_path: Path):
        report = DatasetAuditReport(dataset_root=tmp_path)
        d = report.to_dict()
        for key in ("dataset_root", "n_masks_audited", "n_masks_failed",
                    "n_empty_masks", "n_no_stroma", "overall_class_distribution",
                    "mask_results"):
            assert key in d

    def test_save_creates_json_file(self, tmp_path: Path):
        report = DatasetAuditReport(dataset_root=tmp_path)
        out = tmp_path / "audit.json"
        report.save(out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert "dataset_root" in loaded


# ---------------------------------------------------------------------------
# audit_mask
# ---------------------------------------------------------------------------

class TestAuditMask:
    """Tests for audit_mask()."""

    def test_returns_mask_audit_result(self, tmp_path: Path):
        p = _make_mask(tmp_path, "a.png")
        result = audit_mask(p)
        assert isinstance(result, MaskAuditResult)

    def test_stem_matches_filename(self, tmp_path: Path):
        p = _make_mask(tmp_path, "slide_123.png")
        result = audit_mask(p)
        assert result.stem == "slide_123"

    def test_dimensions_correct(self, tmp_path: Path):
        data = np.zeros((32, 64), dtype=np.uint8)
        img = Image.fromarray(data, mode="L")
        p = tmp_path / "shaped.png"
        img.save(p)
        result = audit_mask(p)
        assert result.height == 32
        assert result.width == 64

    def test_n_pixels_correct(self, tmp_path: Path):
        data = np.zeros((8, 8), dtype=np.uint8)
        img = Image.fromarray(data, mode="L")
        p = tmp_path / "small.png"
        img.save(p)
        result = audit_mask(p)
        assert result.n_pixels == 64

    def test_class_pixel_counts_match(self, tmp_path: Path):
        data = np.array([[0, 0, 1, 1], [2, 2, 1, 0]], dtype=np.uint8)
        img = Image.fromarray(data, mode="L")
        p = tmp_path / "mixed.png"
        img.save(p)
        result = audit_mask(p)
        assert result.class_pixel_counts[0] == 3
        assert result.class_pixel_counts[1] == 3
        assert result.class_pixel_counts[2] == 2

    def test_is_empty_true_for_background_only(self, tmp_path: Path):
        p = _make_empty_mask(tmp_path, "empty.png")
        result = audit_mask(p)
        assert result.is_empty is True

    def test_is_empty_false_for_tissue(self, tmp_path: Path):
        data = np.ones((8, 8), dtype=np.uint8)
        img = Image.fromarray(data, mode="L")
        p = tmp_path / "tissue.png"
        img.save(p)
        result = audit_mask(p)
        assert result.is_empty is False

    def test_has_tumor_true(self, tmp_path: Path):
        p = _make_tumor_only_mask(tmp_path, "tumor.png")
        result = audit_mask(p)
        assert result.has_tumor is True

    def test_has_stroma_true(self, tmp_path: Path):
        p = _make_tumor_stroma_mask(tmp_path, "stroma.png")
        result = audit_mask(p)
        assert result.has_stroma is True

    def test_error_for_corrupt_file(self, tmp_path: Path):
        p = tmp_path / "bad.png"
        p.write_bytes(b"NOT_PNG")
        result = audit_mask(p)
        assert result.error is not None

    def test_fractions_sum_to_one(self, tmp_path: Path):
        data = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        img = Image.fromarray(data, mode="L")
        p = tmp_path / "fracs.png"
        img.save(p)
        result = audit_mask(p)
        total = sum(result.class_pixel_fractions.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_class_distribution
# ---------------------------------------------------------------------------

class TestComputeClassDistribution:
    """Tests for compute_class_distribution()."""

    def _make_result(self, counts: dict, error: str | None = None) -> MaskAuditResult:
        r = MaskAuditResult(stem="test", path=Path("/fake/test.png"))
        r.class_pixel_counts = counts
        r.error = error
        return r

    def test_aggregates_counts_correctly(self):
        r1 = self._make_result({0: 100, 1: 50})
        r2 = self._make_result({0: 50, 2: 100})
        totals, fracs = compute_class_distribution([r1, r2])
        assert totals[0] == 150
        assert totals[1] == 50
        assert totals[2] == 100

    def test_fractions_sum_to_one(self):
        r1 = self._make_result({0: 100, 1: 100, 2: 100})
        _, fracs = compute_class_distribution([r1])
        total = sum(fracs.values())
        assert abs(total - 1.0) < 1e-6

    def test_skips_errored_masks(self):
        r_good = self._make_result({0: 100, 1: 50})
        r_bad = self._make_result({0: 999, 1: 999}, error="corrupt file")
        totals, _ = compute_class_distribution([r_good, r_bad])
        assert totals.get(0, 0) == 100  # Only from good mask

    def test_empty_list_returns_zeros(self):
        totals, fracs = compute_class_distribution([])
        assert all(v == 0 for v in totals.values())


# ---------------------------------------------------------------------------
# detect_empty_masks
# ---------------------------------------------------------------------------

class TestDetectEmptyMasks:
    """Tests for detect_empty_masks()."""

    def test_returns_stems_of_empty_masks(self, tmp_path: Path):
        p1 = _make_empty_mask(tmp_path, "empty1.png")
        p2 = _make_mask(tmp_path, "full.png")
        r1 = audit_mask(p1)
        r2 = audit_mask(p2)
        empty_stems = detect_empty_masks([r1, r2])
        assert "empty1" in empty_stems
        assert "full" not in empty_stems

    def test_returns_empty_list_when_no_empty_masks(self, tmp_path: Path):
        p = _make_mask(tmp_path, "full.png")
        r = audit_mask(p)
        assert detect_empty_masks([r]) == []


# ---------------------------------------------------------------------------
# detect_scorability_issues
# ---------------------------------------------------------------------------

class TestDetectScorabilityIssues:
    """Tests for detect_scorability_issues()."""

    def test_detects_tumor_without_stroma(self, tmp_path: Path):
        p = _make_tumor_only_mask(tmp_path, "tumor_only.png")
        r = audit_mask(p)
        issues = detect_scorability_issues([r])
        assert "tumor_only" in issues

    def test_no_issue_for_tumor_with_stroma(self, tmp_path: Path):
        p = _make_tumor_stroma_mask(tmp_path, "ok.png")
        r = audit_mask(p)
        issues = detect_scorability_issues([r])
        assert "ok" not in issues

    def test_no_issue_for_background_only(self, tmp_path: Path):
        p = _make_empty_mask(tmp_path, "bg.png")
        r = audit_mask(p)
        issues = detect_scorability_issues([r])
        assert "bg" not in issues


# ---------------------------------------------------------------------------
# audit_dataset (orchestrator)
# ---------------------------------------------------------------------------

class TestAuditDataset:
    """Tests for the audit_dataset() orchestrator."""

    def test_returns_dataset_audit_report(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        _make_mask(masks_dir, "s1.png")
        report = audit_dataset(tmp_path)
        assert isinstance(report, DatasetAuditReport)

    def test_correct_n_masks_audited(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        for i in range(3):
            _make_mask(masks_dir, f"s{i}.png")
        report = audit_dataset(tmp_path)
        assert report.n_masks_audited == 3

    def test_detects_empty_masks(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        _make_empty_mask(masks_dir, "empty.png")
        _make_mask(masks_dir, "full.png")
        report = audit_dataset(tmp_path)
        assert report.n_empty_masks == 1

    def test_detects_scorability_issues(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        _make_tumor_only_mask(masks_dir, "tumor_only.png")
        report = audit_dataset(tmp_path)
        assert report.n_no_stroma == 1

    def test_raises_when_masks_dir_missing(self, tmp_path: Path):
        with pytest.raises(DataError, match="Masks directory not found"):
            audit_dataset(tmp_path)

    def test_saves_report_when_path_provided(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        _make_mask(masks_dir, "s1.png")
        out = tmp_path / "audit.json"
        audit_dataset(tmp_path, save_report_to=out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert "n_masks_audited" in loaded
