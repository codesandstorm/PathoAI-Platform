"""
pathoai/wsi/factory.py
======================
Factory for instantiating Whole Slide Image readers.

Decides which reader adapter implementation (e.g. OpenSlideWSI) to create
based on the slide file format and configuration.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 2.1
"""

from __future__ import annotations

from pathlib import Path

from pathoai.core.constants import SUPPORTED_WSI_FORMATS
from pathoai.core.exceptions import WSIReadError
from pathoai.wsi.base import BaseWSI
from pathoai.wsi.openslide_reader import OpenSlideWSI


def get_wsi_reader(path: str | Path) -> BaseWSI:
    """Create and return a WSI reader instance for the given path.

    Checks if the file extension is supported and maps it to the appropriate
    reader implementation.

    Parameters
    ----------
    path : str | Path
        Path to the slide file.

    Returns
    -------
    BaseWSI
        A WSI reader instance implementing the BaseWSI interface.

    Raises
    ------
    WSIReadError
        If the file extension is not supported.
    """
    p = Path(path).resolve()
    ext = p.suffix.lower()

    if ext not in SUPPORTED_WSI_FORMATS:
        raise WSIReadError(
            f"Unsupported WSI file format '{ext}' for file {p}. "
            f"Supported formats: {sorted(SUPPORTED_WSI_FORMATS)}"
        )

    # Currently, all supported formats are handled by OpenSlideWSI
    return OpenSlideWSI(p)
