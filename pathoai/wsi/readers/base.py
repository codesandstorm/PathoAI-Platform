"""
pathoai/wsi/readers/base.py
===========================
Abstract base class definition for Whole Slide Image (WSI) readers.

Provides the unified interface (BaseWSI) that abstracts WSI operations.
This design decouples slide reading backends (e.g. OpenSlide, ASAP, wholeslidedata)
from downstream processing engines (patch extraction, segmentation, etc.).

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import numpy as np


class BaseWSI(ABC):
    """Abstract base class defining the interface for all Whole Slide Image readers.

    Implementations must support the context manager protocol (`__enter__` and
    `__exit__`) to guarantee resources are cleaned up cleanly.
    """

    @abstractmethod
    def __enter__(self) -> BaseWSI:
        """Enter the context manager, opening the WSI if not already open."""
        pass

    @abstractmethod
    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Exit the context manager, guaranteeing slide closure."""
        pass

    @abstractmethod
    def open(self) -> None:
        """Open the slide file and load necessary system/library resources.

        If the slide is already open, this is a no-op.

        Raises
        ------
        WSIReadError
            If the slide file does not exist, cannot be opened, or is corrupt.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the slide file and release all associated system/library resources.

        If the slide is already closed, this is a no-op.
        """
        pass

    @abstractmethod
    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Read a rectangular region from the slide and return it as an RGB uint8 array.

        Parameters
        ----------
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner in the level 0 reference frame.
        level : int
            Pyramid level to read the region from.
        size : Tuple[int, int]
            (width, height) of the region to read *at the specified level*.

        Returns
        -------
        np.ndarray
            NumPy array of shape (height, width, 3), dtype uint8, containing the RGB image.

        Raises
        ------
        WSIReadError
            If the region is out of slide bounds, the level is invalid, or reading fails.
        """
        pass

    @property
    @abstractmethod
    def path(self) -> Path:
        """Absolute Path to the slide file."""
        pass

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """True if the slide resource is open, False otherwise."""
        pass

    @property
    @abstractmethod
    def properties(self) -> dict[str, str]:
        """Raw format-specific slide properties."""
        pass

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Number of pyramid levels in the image."""
        pass

    @property
    @abstractmethod
    def level_dimensions(self) -> list[tuple[int, int]]:
        """List of (width, height) pixel dimensions for each pyramid level."""
        pass

    @property
    @abstractmethod
    def level_downsamples(self) -> list[float]:
        """List of downsample factors for each pyramid level."""
        pass

    @property
    @abstractmethod
    def associated_images(self) -> list[str]:
        """Keys of associated images (e.g. 'label', 'macro', 'thumbnail') available."""
        pass
