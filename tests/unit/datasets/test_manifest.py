"""
tests/unit/datasets/test_manifest.py
====================================
Unit tests for the Manifest Generator.

Tests cover:
- generate_dataset_manifest slides discovery and directory checks
- Mocked slide profiling, tissue masking, and patch sampling
- Mask class pixel distribution calculation
- JSON file serialization output

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pathoai.core.exceptions import DataError
from pathoai.datasets.manifest import generate_dataset_manifest
from pathoai.wsi.metadata.metadata import SlideMetadata
from pathoai.wsi.patches.patches import PatchMetadata


class TestManifestGenerator:
    """Verifies end-to-end manifest building with fully mocked WSI dependencies."""

    @pytest.fixture()
    def fake_dataset_structure(self, tmp_path: Path):
        """Creates dummy directory structure and empty slide/mask files."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Create two fake slides and one matching mask
        (images_dir / "slide_001.tif").write_bytes(b"dummy")
        (images_dir / "slide_002.svs").write_bytes(b"dummy")
        (masks_dir / "slide_001.png").write_bytes(b"dummy_mask")

        return tmp_path

    @patch("pathoai.datasets.manifest.get_wsi_reader")
    @patch("pathoai.datasets.manifest.extract_metadata")
    @patch("pathoai.datasets.manifest.get_slide_thumbnail")
    @patch("pathoai.datasets.manifest.TissueDetector")
    @patch("pathoai.datasets.manifest.PatchSampler")
    @patch("PIL.Image.open")
    def test_generate_manifest_success(
        self,
        mock_image_open,
        mock_sampler_class,
        mock_tissue_detector_class,
        mock_get_thumbnail,
        mock_extract_meta,
        mock_get_reader,
        fake_dataset_structure,
    ):
        # 1. Setup mocks
        mock_reader = MagicMock()
        mock_get_reader.return_value = mock_reader

        mock_meta = SlideMetadata(
            path=Path(fake_dataset_structure / "images/slide_001.tif"),
            vendor="generic",
            dimensions=(4000, 3000),
            level_count=1,
            level_dimensions=[(4000, 3000)],
            level_downsamples=[1.0],
            mpp_x=0.50,
            mpp_y=0.50,
            magnification=20.0,
            associated_images=[],
            properties={},
        )
        mock_extract_meta.return_value = mock_meta

        # Thumbnail & tissue detector
        mock_get_thumbnail.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_detector.detect_tissue.return_value = (np.ones((100, 100), dtype=np.uint8), 0.5)
        mock_tissue_detector_class.return_value = mock_detector

        # Sampler generates 1 patch
        mock_sampler = MagicMock()
        mock_sampler.sample_patches.return_value = [
            PatchMetadata(
                slide_path=mock_meta.path,
                x_level0=1000,
                y_level0=1000,
                patch_size=512,
                target_mpp=0.50,
                tissue_coverage=0.9,
            )
        ]
        mock_sampler_class.return_value = mock_sampler

        # Mock PIL image for mask loading
        mock_mask_img = MagicMock(spec=Image.Image)
        mock_mask_img.size = (400, 300)
        # crop returns a sub-image representing patch mask
        mock_crop_img = MagicMock(spec=Image.Image)
        mock_crop_img.size = (51, 51)
        mock_mask_img.crop.return_value = mock_crop_img
        mock_image_open.return_value = mock_mask_img

        # 2. Run generator
        output_json = fake_dataset_structure / "manifest.json"
        entries = generate_dataset_manifest(
            dataset_root=fake_dataset_structure,
            output_path=output_json,
        )

        # 3. Assertions
        # Both slides were processed (slide_001 and slide_002)
        assert len(entries) == 2
        assert output_json.exists()

        # Check first entry fields
        first = entries[0]
        assert "slide_path" in first
        assert "mask_path" in first
        assert first["x_level0"] == 1000
        assert first["tissue_coverage"] == 0.9

        # Ensure JSON is readable
        loaded = json.loads(output_json.read_text())
        assert len(loaded) == 2
        assert loaded[0]["x_level0"] == 1000

    def test_raises_on_invalid_dataset_root(self, tmp_path: Path):
        with pytest.raises(DataError, match="Slides directory not found"):
            generate_dataset_manifest(tmp_path / "nonexistent")
