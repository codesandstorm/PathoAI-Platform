"""
tests/conftest.py
=================
Shared pytest fixtures for PathoAI-Platform test suite.

All fixtures here are available to unit and integration tests via dependency injection.
Test data uses ONLY synthetic generators — no real WSI files are required.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from pathoai.core.config import ConfigManager
from pathoai.core.logger import configure_logging


# ---------------------------------------------------------------------------
# LOGGING FIXTURE — configure logging for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def configure_test_logging(tmp_path: Path) -> None:
    """Configure minimal logging for each test (autouse = applied to all tests)."""
    configure_logging(
        log_dir=None,   # No file logging in tests — console only
        level="WARNING",  # Suppress INFO logs in tests; keep WARNING/ERROR visible
        console=False,    # No console spam during test runs
    )
    yield
    # Teardown: reset ConfigManager after each test that may have initialized it
    ConfigManager.reset()


# ---------------------------------------------------------------------------
# CONFIG FIXTURES
# ---------------------------------------------------------------------------

TEST_CONFIG_DIR = Path(__file__).parent / "fixtures" / "configs"


@pytest.fixture
def test_config_path() -> Path:
    """Path to the minimal test configuration YAML file."""
    return TEST_CONFIG_DIR / "test_base.yaml"


@pytest.fixture
def base_config(test_config_path: Path):
    """Initialize and return the test configuration as a ConfigNode."""
    return ConfigManager.initialize(base_config=test_config_path, apply_env_vars=False)


# ---------------------------------------------------------------------------
# SYNTHETIC IMAGE FIXTURES
# ---------------------------------------------------------------------------

def make_synthetic_hne_thumbnail(
    size: tuple = (256, 256),
    tissue_fraction: float = 0.6,
) -> np.ndarray:
    """Generate a synthetic H&E-like thumbnail for tissue detection tests.

    Parameters
    ----------
    size : tuple
        (width, height) of the thumbnail in pixels.
    tissue_fraction : float
        Fraction of the image that should be colored tissue.

    Returns
    -------
    np.ndarray
        RGB thumbnail array. Shape: (H, W, 3), uint8.
    """
    h, w = size[1], size[0]
    image = np.full((h, w, 3), 245, dtype=np.uint8)  # Start white (background)

    # Add tissue region (pink-purple H&E appearance)
    tissue_h = int(h * tissue_fraction)
    tissue_w = int(w * tissue_fraction)
    tissue_top = (h - tissue_h) // 2
    tissue_left = (w - tissue_w) // 2

    rng = np.random.RandomState(seed=42)
    tissue_color = rng.randint(140, 200, size=(tissue_h, tissue_w, 3), dtype=np.uint8)
    # Bias toward purple-pink (H&E: high R, low G, high B)
    tissue_color[:, :, 0] = np.clip(tissue_color[:, :, 0] + 40, 0, 255)  # More red
    tissue_color[:, :, 2] = np.clip(tissue_color[:, :, 2] + 30, 0, 255)  # More blue
    tissue_color[:, :, 1] = np.clip(tissue_color[:, :, 1] - 30, 0, 255)  # Less green

    image[tissue_top:tissue_top + tissue_h, tissue_left:tissue_left + tissue_w] = tissue_color
    return image


def make_synthetic_patch(
    size: int = 512,
    n_channels: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic H&E-like patch for model input tests.

    Parameters
    ----------
    size : int
        Width = height of the square patch.
    n_channels : int
        Number of color channels (3 for RGB).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        RGB patch. Shape: (size, size, 3), uint8.
    """
    rng = np.random.RandomState(seed=seed)
    # Generate low-frequency tissue-like texture
    base = rng.randint(150, 220, size=(size, size, n_channels), dtype=np.uint8)
    noise = rng.randint(0, 30, size=(size, size, n_channels), dtype=np.uint8)
    patch = np.clip(base.astype(np.int32) - noise.astype(np.int32), 0, 255).astype(np.uint8)
    return patch


def make_synthetic_segmentation_mask(
    size: int = 512,
    n_classes: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic segmentation prediction mask.

    Parameters
    ----------
    size : int
        Width = height of the square mask.
    n_classes : int
        Number of tissue classes.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Integer class label per pixel. Shape: (size, size), uint8.
        Values in range [0, n_classes-1].
    """
    rng = np.random.RandomState(seed=seed)
    # Create spatially coherent regions using simple block assignment
    mask = np.zeros((size, size), dtype=np.uint8)
    block_size = size // 4
    class_id = 0
    for row in range(4):
        for col in range(4):
            class_id = (class_id + 1) % n_classes
            r_start = row * block_size
            c_start = col * block_size
            mask[r_start:r_start + block_size, c_start:c_start + block_size] = class_id
    return mask


def make_synthetic_bboxes(
    n_cells: int = 20,
    image_size: int = 256,
    cell_size: int = 20,
    seed: int = 42,
) -> tuple:
    """Generate synthetic bounding boxes and labels for detection tests.

    Parameters
    ----------
    n_cells : int
        Number of cells to generate.
    image_size : int
        Image width = height.
    cell_size : int
        Approximate cell diameter in pixels.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (boxes, labels, scores) as numpy arrays.
        boxes: (N, 4) [x1, y1, x2, y2], float32
        labels: (N,) int64
        scores: (N,) float32
    """
    rng = np.random.RandomState(seed=seed)
    n_classes = 3  # background excluded; classes 1, 2, 3

    cx = rng.randint(cell_size, image_size - cell_size, size=n_cells)
    cy = rng.randint(cell_size, image_size - cell_size, size=n_cells)
    half = cell_size // 2

    boxes = np.stack([
        cx - half, cy - half, cx + half, cy + half,
    ], axis=1).astype(np.float32)

    labels = rng.randint(1, n_classes + 1, size=n_cells).astype(np.int64)
    scores = rng.uniform(0.5, 1.0, size=n_cells).astype(np.float32)

    return boxes, labels, scores


# ---------------------------------------------------------------------------
# PATH FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory for each test."""
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
