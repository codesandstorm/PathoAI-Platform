"""
pathoai/wsi/readers/openslide_reader.py
======================================
OpenSlide adapter implementation of BaseWSI.

Wraps the OpenSlide C library to read Whole Slide Images (WSIs),
performing format abstraction, resource management, exception wrapping,
input validation, and converting RGBA PIL images to standard RGB NumPy arrays.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np

from pathoai.core.exceptions import WSIReadError
from pathoai.core.logger import get_logger
from pathoai.wsi.readers.base import BaseWSI

logger = get_logger(__name__)


class OpenSlideWSI(BaseWSI):
    """BaseWSI adapter wrapping the OpenSlide library.

    Manages the lifecycle of an open slide file and provides safe,
    validated access to image regions, mapping internal OpenSlide/C library
    errors to `WSIReadError`.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Parameters
        ----------
        path : str | Path
            Path to the slide file.
        """
        self._path = Path(path).resolve()
        self._slide: Any = None

    @property
    def path(self) -> Path:
        """Absolute Path to the slide file."""
        return self._path

    @property
    def is_open(self) -> bool:
        """True if the slide resource is open, False otherwise."""
        return self._slide is not None

    def __enter__(self) -> OpenSlideWSI:
        """Enter the context manager, opening the slide."""
        self.open()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Exit the context manager, closing the slide."""
        self.close()

    def open(self) -> None:
        """Open the slide file and load the OpenSlide object.

        If the slide is already open, this is a no-op.

        Raises
        ------
        WSIReadError
            If the slide file is missing, corrupt, or OpenSlide cannot load it.
        """
        if self.is_open:
            return

        logger.debug("Opening slide: %s", self._path)
        if not self._path.exists():
            raise WSIReadError(f"Slide file does not exist: {self._path}")

        try:
            import openslide
        except (ImportError, OSError) as exc:
            raise WSIReadError(
                f"Failed to import/load OpenSlide: {exc}. "
                "Ensure OpenSlide C binaries are installed and added to the system PATH."
            ) from exc

        try:
            self._slide = openslide.OpenSlide(str(self._path))
        except Exception as exc:
            raise WSIReadError(
                f"OpenSlide failed to open slide {self._path}: {exc}"
            ) from exc

    def close(self) -> None:
        """Close the slide file and release OpenSlide resources.

        If the slide is already closed, this is a no-op.
        """
        if not self.is_open:
            return

        logger.debug("Closing slide: %s", self._path)
        try:
            self._slide.close()
        except Exception as exc:
            logger.warning("Error closing OpenSlide instance for %s: %s", self._path, exc)
        finally:
            self._slide = None

    def _check_open(self) -> None:
        """Raise WSIReadError if the slide is not open."""
        if not self.is_open:
            raise WSIReadError(f"Slide is not open: {self._path}")

    @property
    def properties(self) -> dict[str, str]:
        """Raw format-specific slide properties."""
        self._check_open()
        return dict(self._slide.properties)

    @property
    def level_count(self) -> int:
        """Number of pyramid levels in the image."""
        self._check_open()
        return int(self._slide.level_count)

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        """List of (width, height) pixel dimensions for each pyramid level."""
        self._check_open()
        return list(self._slide.level_dimensions)

    @property
    def level_downsamples(self) -> list[float]:
        """List of downsample factors for each pyramid level."""
        self._check_open()
        return [float(ds) for ds in self._slide.level_downsamples]

    @property
    def associated_images(self) -> list[str]:
        """Keys of associated images available."""
        self._check_open()
        return list(self._slide.associated_images.keys())

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
            If the slide is not open, level is out of bounds, coordinates or size are
            invalid, or the underlying library read operation fails.
        """
        self._check_open()

        x, y = location
        w, h = size

        if x < 0 or y < 0:
            raise WSIReadError(
                f"Coordinates must be non-negative. Got location: ({x}, {y})"
            )
        if w <= 0 or h <= 0:
            raise WSIReadError(
                f"Region dimensions must be positive. Got size: ({w}, {h})"
            )
        if level < 0:
            raise WSIReadError(
                f"Pyramid level must be non-negative. Got level: {level}"
            )

        try:
            level_count = self._slide.level_count
        except Exception as exc:
            raise WSIReadError(
                f"Failed to query level count for slide {self._path}: {exc}"
            ) from exc

        if level >= level_count:
            raise WSIReadError(
                f"Requested level {level} exceeds slide level count of {level_count}"
            )

        try:
            # OpenSlide read_region returns a PIL Image in RGBA mode
            pil_img = self._slide.read_region((x, y), level, (w, h))
            rgba_arr = np.array(pil_img, dtype=np.uint8)

            # Convert RGBA to standard RGB
            if rgba_arr.ndim == 3 and rgba_arr.shape[2] == 4:
                return rgba_arr[..., :3]
            elif rgba_arr.ndim == 3 and rgba_arr.shape[2] == 3:
                return rgba_arr
            else:
                raise WSIReadError(
                    f"Unexpected image shape returned from OpenSlide: {rgba_arr.shape}"
                )
        except Exception as exc:
            raise WSIReadError(
                f"Failed to read region at location={location}, level={level}, size={size} "
                f"on slide {self._path}: {exc}"
            ) from exc
