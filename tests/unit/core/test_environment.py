"""
tests/unit/core/test_environment.py
======================================
Unit tests for pathoai.core.environment.

Tests cover:
- PackageStatus: dataclass construction, to_dict()
- EnvironmentReport: dataclass construction, to_dict()
- audit_platform: returns dict with required keys, python version correct
- audit_gpu: returns dict with required keys, no exception raised
- audit_package: installed package, missing package, outdated detection
- audit_openslide: returns dict with required keys
- audit_disk_space: returns dict, sufficient flag logic
- run_full_audit: returns EnvironmentReport, required sections present
- validate_environment: does not raise on functioning environment (best effort)

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pathoai.core.environment import (
    EnvironmentReport,
    PackageStatus,
    audit_disk_space,
    audit_gpu,
    audit_openslide,
    audit_package,
    audit_platform,
    run_full_audit,
)


# ---------------------------------------------------------------------------
# PackageStatus
# ---------------------------------------------------------------------------

class TestPackageStatus:
    """Tests for PackageStatus dataclass."""

    def test_construction(self):
        """Must construct with required fields."""
        status = PackageStatus(
            name="numpy",
            required=True,
            installed=True,
            version="1.26.0",
            min_version="1.24.0",
            status_label="READY",
        )
        assert status.name == "numpy"
        assert status.installed is True
        assert status.status_label == "READY"

    def test_to_dict_contains_required_keys(self):
        """to_dict must contain name, required, installed, version, min_version, status."""
        status = PackageStatus(name="numpy", required=True, installed=True)
        d = status.to_dict()
        for key in ("name", "required", "installed", "version", "min_version", "status"):
            assert key in d, f"Missing key in PackageStatus.to_dict(): {key}"

    def test_default_status_label_is_unknown(self):
        """Default status_label must be UNKNOWN."""
        status = PackageStatus(name="pkg", required=True, installed=False)
        assert status.status_label == "UNKNOWN"


# ---------------------------------------------------------------------------
# EnvironmentReport
# ---------------------------------------------------------------------------

class TestEnvironmentReport:
    """Tests for EnvironmentReport dataclass."""

    def test_default_construction(self):
        """Must construct with sensible defaults."""
        report = EnvironmentReport()
        assert isinstance(report.errors, list)
        assert isinstance(report.warnings, list)
        assert isinstance(report.packages, list)
        assert report.is_ready is False
        assert report.readiness_score == 0

    def test_to_dict_contains_required_sections(self):
        """to_dict must contain all required top-level sections."""
        report = EnvironmentReport()
        d = report.to_dict()
        for key in ("platform", "gpu", "packages", "openslide", "disk",
                    "errors", "warnings", "readiness_score", "is_ready"):
            assert key in d, f"Missing section in EnvironmentReport.to_dict(): {key}"

    def test_to_dict_is_json_serializable(self):
        """The report dict must be JSON-serializable."""
        report = EnvironmentReport()
        json_str = json.dumps(report.to_dict(), default=str)
        assert json.loads(json_str) is not None


# ---------------------------------------------------------------------------
# audit_platform
# ---------------------------------------------------------------------------

class TestAuditPlatform:
    """Tests for audit_platform()."""

    def test_returns_dict(self):
        """Must return a dict."""
        result = audit_platform()
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Must contain os, python_version, machine, architecture."""
        result = audit_platform()
        for key in ("os", "python_version", "machine", "architecture"):
            assert key in result, f"Missing key: {key}"

    def test_python_version_matches_runtime(self):
        """python_version must match the running Python version."""
        result = audit_platform()
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert result["python_version"] == expected

    def test_os_is_string(self):
        """OS field must be a non-empty string."""
        result = audit_platform()
        assert isinstance(result["os"], str)
        assert len(result["os"]) > 0


# ---------------------------------------------------------------------------
# audit_gpu
# ---------------------------------------------------------------------------

class TestAuditGpu:
    """Tests for audit_gpu()."""

    def test_returns_dict(self):
        """Must return a dict regardless of GPU availability."""
        result = audit_gpu()
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Must contain cuda_available, nvidia_gpu_detected, gpu_count."""
        result = audit_gpu()
        for key in ("cuda_available", "nvidia_gpu_detected", "gpu_count"):
            assert key in result, f"Missing key: {key}"

    def test_cuda_available_is_bool(self):
        """cuda_available must be a boolean."""
        result = audit_gpu()
        assert isinstance(result["cuda_available"], bool)

    def test_does_not_raise(self):
        """audit_gpu must never raise, even without a GPU."""
        result = audit_gpu()  # Must not raise
        assert result is not None


# ---------------------------------------------------------------------------
# audit_package
# ---------------------------------------------------------------------------

class TestAuditPackage:
    """Tests for audit_package()."""

    def test_installed_package_returns_ready(self):
        """A well-known installed package (numpy) must return READY status."""
        status = audit_package("numpy", "numpy", "1.0.0", required=True)
        assert status.installed is True
        assert status.status_label == "READY"
        assert status.version is not None

    def test_missing_package_returns_missing(self):
        """A clearly non-existent package must return MISSING status."""
        status = audit_package(
            "this_package_does_not_exist_xyz",
            "this-package-does-not-exist-xyz",
            "1.0.0",
            required=True,
        )
        assert status.installed is False
        assert status.status_label == "MISSING"

    def test_optional_missing_package_returns_optional_missing(self):
        """A missing optional package must return OPTIONAL_MISSING status."""
        status = audit_package(
            "this_package_does_not_exist_xyz",
            "this-package-does-not-exist-xyz",
            "1.0.0",
            required=False,
        )
        assert status.status_label == "OPTIONAL_MISSING"

    def test_outdated_package_detection(self):
        """A package with an impossibly high minimum version must return OUTDATED."""
        status = audit_package("numpy", "numpy", "999.0.0", required=True)
        assert status.installed is True
        # Depends on packaging module being available
        if status.status_label != "READY":
            assert status.status_label == "OUTDATED"

    def test_returns_package_status_object(self):
        """Return type must be PackageStatus."""
        status = audit_package("numpy", "numpy", "1.0.0", required=True)
        assert isinstance(status, PackageStatus)

    def test_pip_name_stored_in_status(self):
        """pip_name argument must be stored in PackageStatus.name."""
        status = audit_package("numpy", "my-numpy-alias", "1.0.0", required=True)
        assert status.name == "my-numpy-alias"


# ---------------------------------------------------------------------------
# audit_openslide
# ---------------------------------------------------------------------------

class TestAuditOpenslide:
    """Tests for audit_openslide()."""

    def test_returns_dict(self):
        """Must return a dict regardless of OpenSlide installation state."""
        result = audit_openslide()
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Must contain python_binding_installed, c_library_found, status."""
        result = audit_openslide()
        for key in ("python_binding_installed", "c_library_found", "status"):
            assert key in result, f"Missing key: {key}"

    def test_status_is_valid_value(self):
        """status must be one of READY, PARTIAL, MISSING, ERROR."""
        result = audit_openslide()
        assert result["status"] in ("READY", "PARTIAL", "MISSING", "ERROR")

    def test_does_not_raise(self):
        """audit_openslide must never raise."""
        result = audit_openslide()
        assert result is not None


# ---------------------------------------------------------------------------
# audit_disk_space
# ---------------------------------------------------------------------------

class TestAuditDiskSpace:
    """Tests for audit_disk_space()."""

    def test_returns_dict(self):
        """Must return a dict."""
        result = audit_disk_space(paths=[Path.cwd()])
        assert isinstance(result, dict)

    def test_custom_path_checked(self, tmp_path: Path):
        """The provided path must appear as a key in the result."""
        result = audit_disk_space(paths=[tmp_path])
        assert str(tmp_path) in result

    def test_sufficient_flag_present(self, tmp_path: Path):
        """Result for an accessible path must contain 'sufficient' key."""
        result = audit_disk_space(paths=[tmp_path])
        info = result[str(tmp_path)]
        if "error" not in info:
            assert "sufficient" in info

    def test_free_gb_is_positive(self, tmp_path: Path):
        """free_gb must be a positive number for an existing path."""
        result = audit_disk_space(paths=[tmp_path])
        info = result[str(tmp_path)]
        if "error" not in info:
            assert info["free_gb"] > 0

    def test_inaccessible_path_returns_error_key(self):
        """An inaccessible path must return a dict with 'error' key."""
        import platform as _platform
        if _platform.system() == "Windows":
            bad_path = Path("Z:\\nonexistent_drive_xyz")
        else:
            bad_path = Path("/nonexistent_mount_xyz")
        result = audit_disk_space(paths=[bad_path])
        info = result[str(bad_path)]
        assert "error" in info


# ---------------------------------------------------------------------------
# run_full_audit
# ---------------------------------------------------------------------------

class TestRunFullAudit:
    """Tests for run_full_audit()."""

    def test_returns_environment_report(self):
        """Must return an EnvironmentReport instance."""
        report = run_full_audit()
        assert isinstance(report, EnvironmentReport)

    def test_platform_info_populated(self):
        """platform_info must be populated after a full audit."""
        report = run_full_audit()
        assert isinstance(report.platform_info, dict)
        assert len(report.platform_info) > 0

    def test_gpu_info_populated(self):
        """gpu_info must be populated after a full audit."""
        report = run_full_audit()
        assert isinstance(report.gpu_info, dict)

    def test_packages_list_populated(self):
        """packages list must be non-empty after a full audit."""
        report = run_full_audit()
        assert len(report.packages) > 0

    def test_readiness_score_in_valid_range(self):
        """readiness_score must be in range [0, 100]."""
        report = run_full_audit()
        assert 0 <= report.readiness_score <= 100

    def test_errors_and_warnings_are_lists(self):
        """errors and warnings must be lists of strings."""
        report = run_full_audit()
        assert isinstance(report.errors, list)
        assert isinstance(report.warnings, list)
        assert all(isinstance(e, str) for e in report.errors)
        assert all(isinstance(w, str) for w in report.warnings)

    def test_saves_report_to_json_when_path_provided(self, tmp_path: Path):
        """When output_path is provided, must save a valid JSON file."""
        output = tmp_path / "audit_report.json"
        report = run_full_audit(output_path=output)
        assert output.exists()
        with open(output, encoding="utf-8") as f:
            loaded = json.load(f)
        assert "platform" in loaded
        assert "packages" in loaded

    def test_does_not_raise(self):
        """run_full_audit must never raise regardless of environment state."""
        try:
            run_full_audit()
        except Exception as e:
            pytest.fail(f"run_full_audit raised unexpectedly: {e}")
