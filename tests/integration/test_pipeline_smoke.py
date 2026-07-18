"""
tests/integration/test_pipeline_smoke.py
==========================================
Integration smoke tests for the PathoAI-Platform core pipeline.

These tests exercise end-to-end flows using ONLY synthetic data — no real WSI
files, no GPU, and no trained model weights are required. They verify that
core components wire together correctly without runtime errors.

Smoke tests are intentionally lightweight; they verify integration contracts,
not algorithm correctness (which is covered by unit tests).

Skipped in CI if optional dependencies (torch, openslide) are absent.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.9
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(root: Path, n_slides: int = 2) -> Path:
    """Create a minimal synthetic TIGER-style dataset for integration tests."""
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_slides):
        # Fake WSI (just a byte string — we won't open it with OpenSlide)
        (images_dir / f"slide_{i:03d}.tif").write_bytes(b"TIFF_DUMMY_DATA" * 100)

        # Valid grayscale PNG mask with class IDs 0–5
        data = np.zeros((64, 64), dtype=np.uint8)
        data[:16, :] = 1   # tumor
        data[16:32, :] = 2  # stroma
        data[32:48, :] = 3  # lymphocytes
        Image.fromarray(data, mode="L").save(masks_dir / f"slide_{i:03d}.png")

    return root


# ---------------------------------------------------------------------------
# Smoke Test 1: Core package imports without error
# ---------------------------------------------------------------------------

class TestCoreImports:
    """Verify all core modules import correctly in integration."""

    def test_pathoai_root_imports(self):
        """pathoai package must import without error."""
        import pathoai
        assert hasattr(pathoai, "__version__")

    def test_core_constants_import(self):
        from pathoai.core import constants
        assert constants.N_TISSUE_CLASSES == 6

    def test_core_exceptions_import(self):
        from pathoai.core.exceptions import PathoAIException, DatasetValidationError
        assert issubclass(DatasetValidationError, PathoAIException)

    def test_core_config_import(self):
        from pathoai.core.config import ConfigManager, ConfigNode
        assert ConfigManager is not None

    def test_core_path_utils_import(self):
        from pathoai.core.utils.path_utils import ensure_directory, resolve_path
        assert callable(ensure_directory)
        assert callable(resolve_path)

    def test_validation_import(self):
        from pathoai.validation import validate_dataset, audit_dataset
        assert callable(validate_dataset)
        assert callable(audit_dataset)

    def test_visualization_import(self):
        from pathoai.visualization import colorize_mask, blend_mask_overlay
        assert callable(colorize_mask)
        assert callable(blend_mask_overlay)


# ---------------------------------------------------------------------------
# Smoke Test 2: Configuration system end-to-end
# ---------------------------------------------------------------------------

class TestConfigurationSmoke:
    """Verify configuration loading and access."""

    def test_load_config_from_base_yaml(self):
        """config/base.yaml must load successfully."""
        from pathoai.core.config import ConfigManager, get_config
        ConfigManager.reset()
        ConfigManager.initialize(
            base_config=Path("config/base.yaml"),
            apply_env_vars=False,
        )
        cfg = get_config()
        assert cfg.pipeline.seed == 42
        assert cfg.pipeline.device in ("auto", "cpu", "cuda")
        ConfigManager.reset()

    def test_override_file_takes_priority(self, tmp_path: Path):
        """Override YAML must take priority over base.yaml."""
        from pathoai.core.config import ConfigManager
        override = tmp_path / "override.yaml"
        override.write_text("pipeline:\n  seed: 9999\n", encoding="utf-8")
        ConfigManager.reset()
        ConfigManager.initialize(
            base_config=Path("config/base.yaml"),
            overrides=[override],
            apply_env_vars=False,
        )
        cfg = ConfigManager.get_instance()
        assert cfg.pipeline.seed == 9999
        ConfigManager.reset()

    def test_config_hash_is_stable(self):
        """Same config file must produce the same hash on repeated loads."""
        from pathoai.core.config import ConfigManager
        ConfigManager.reset()
        ConfigManager.initialize(
            base_config=Path("config/base.yaml"),
            apply_env_vars=False,
        )
        h1 = ConfigManager.get_config_hash()
        ConfigManager.reset()
        ConfigManager.initialize(
            base_config=Path("config/base.yaml"),
            apply_env_vars=False,
        )
        h2 = ConfigManager.get_config_hash()
        assert h1 == h2
        ConfigManager.reset()


# ---------------------------------------------------------------------------
# Smoke Test 3: Dataset validation pipeline
# ---------------------------------------------------------------------------

class TestDatasetValidationSmoke:
    """Verify validate_dataset → audit_dataset integration."""

    def test_validation_followed_by_audit(self, tmp_path: Path):
        """validate_dataset then audit_dataset must both complete without error."""
        from pathoai.validation import validate_dataset, audit_dataset

        _make_synthetic_dataset(tmp_path, n_slides=3)

        val_report = validate_dataset(
            tmp_path,
            check_image_integrity=True,
            check_duplicates=False,
        )
        assert val_report.is_valid is True
        assert val_report.n_slides_found == 3

        audit_report = audit_dataset(tmp_path)
        assert audit_report.n_masks_audited == 3
        assert audit_report.n_masks_failed == 0

    def test_validation_report_is_json_serializable(self, tmp_path: Path):
        """validate_dataset output must serialize to valid JSON."""
        from pathoai.validation import validate_dataset

        _make_synthetic_dataset(tmp_path)
        report = validate_dataset(tmp_path, check_image_integrity=False)
        json_str = json.dumps(report.to_dict(), default=str)
        assert json.loads(json_str)["is_valid"] is True

    def test_audit_detects_empty_mask(self, tmp_path: Path):
        """audit_dataset must detect an empty mask in a mixed dataset."""
        from pathoai.validation import audit_dataset

        _make_synthetic_dataset(tmp_path, n_slides=2)
        # Add an empty mask
        empty = np.zeros((64, 64), dtype=np.uint8)
        Image.fromarray(empty, mode="L").save(tmp_path / "masks" / "empty_slide.png")

        report = audit_dataset(tmp_path)
        assert report.n_empty_masks >= 1


# ---------------------------------------------------------------------------
# Smoke Test 4: Visualization pipeline
# ---------------------------------------------------------------------------

class TestVisualizationSmoke:
    """Verify colormap → LUT → overlay integration."""

    def test_colorize_then_overlay_pipeline(self):
        """colorize_mask → blend_mask_overlay must produce valid output."""
        from pathoai.visualization import build_tissue_lut, blend_mask_overlay

        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        mask = np.random.randint(0, 6, (128, 128), dtype=np.uint8)
        lut = build_tissue_lut(alpha=180)

        result = blend_mask_overlay(image, mask, alpha=0.4, lut=lut)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8
        assert 0 <= result.min()
        assert result.max() <= 255

    def test_patch_grid_from_synthetic_patches(self):
        """make_patch_grid must produce correct grid from random patches."""
        from pathoai.visualization import make_patch_grid

        patches = [
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            for _ in range(16)
        ]
        grid = make_patch_grid(patches, n_cols=4, pad=2)
        assert grid.ndim == 3
        assert grid.shape[2] == 3

    def test_class_color_legend_matches_constants(self):
        """Legend entries must match the constants-defined tissue classes."""
        from pathoai.visualization import get_class_color_legend
        from pathoai.core.constants import TISSUE_CLASSES

        legend = get_class_color_legend()
        legend_names = {entry["class_name"] for entry in legend}
        expected_names = set(TISSUE_CLASSES.values())
        assert legend_names == expected_names


# ---------------------------------------------------------------------------
# Smoke Test 5: Reproducibility integration
# ---------------------------------------------------------------------------

class TestReproducibilitySmoke:
    """Verify reproducibility module integration."""

    def test_set_seed_then_capture_snapshot(self, tmp_path: Path):
        """set_global_seed → capture_environment_snapshot → save must not raise."""
        from pathoai.core.reproducibility import (
            set_global_seed,
            capture_environment_snapshot,
            save_environment_snapshot,
        )
        set_global_seed(42)
        snapshot = capture_environment_snapshot(
            experiment_id="integration_smoke_001",
            config_hash="abc123def456",
        )
        output = save_environment_snapshot(snapshot, output_dir=tmp_path)
        assert output.exists()
        loaded = json.loads(output.read_text())
        assert loaded["experiment_id"] == "integration_smoke_001"
        assert loaded["python"]["version"].startswith("3.")

    def test_seeds_are_deterministic_across_calls(self):
        """Two identical seeds must produce identical random arrays."""
        from pathoai.core.reproducibility import set_global_seed

        set_global_seed(42)
        arr1 = np.random.rand(10)

        set_global_seed(42)
        arr2 = np.random.rand(10)

        np.testing.assert_array_equal(arr1, arr2)


# ---------------------------------------------------------------------------
# Smoke Test 6: Path utilities integration
# ---------------------------------------------------------------------------

class TestPathUtilsSmoke:
    """Verify path_utils creates and validates project structure."""

    def test_create_then_validate_project_structure(self, tmp_path: Path):
        """create_project_structure then validate_project_structure must report 0 missing."""
        from pathoai.core.utils.path_utils import (
            create_project_structure,
            validate_project_structure,
        )
        create_project_structure(tmp_path)
        missing = validate_project_structure(tmp_path)
        assert missing == [], f"Missing directories after create_project_structure: {missing}"

    def test_safe_copy_then_verify_file(self, tmp_path: Path):
        """safe_copy_file then ensure_file_exists must succeed."""
        from pathoai.core.utils.path_utils import safe_copy_file, ensure_file_exists

        src = tmp_path / "source.json"
        src.write_text('{"key": "value"}')
        dst = tmp_path / "output" / "copy.json"

        copied = safe_copy_file(src, dst, overwrite=False)
        verified = ensure_file_exists(copied)
        assert verified == copied
