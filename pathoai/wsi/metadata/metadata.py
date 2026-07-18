"""
pathoai/wsi/metadata/metadata.py
================================
WSI metadata representation and extraction engine.

Extracts scanner-independent, standardized SlideMetadata from a BaseWSI reader.
Handles vendor-specific properties (MPP, objective power, vendor detection)
and validates slide integrity.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pathoai.core.exceptions import MetadataExtractionError
from pathoai.core.logger import get_logger
from pathoai.wsi.readers.base import BaseWSI

logger = get_logger(__name__)


@dataclass(frozen=True)
class SlideMetadata:
    """Canonical representation of WSI metadata.

    Downstream processing stages (pyramid, patch sampling, models) query this
    dataclass rather than interacting directly with format-specific libraries.

    Attributes:
        path: Absolute Path to the slide file.
        vendor: Scanner vendor name (e.g. 'aperio', 'hamamatsu', 'leica', 'generic').
        dimensions: Slide pixel dimensions (width, height) at level 0 (full resolution).
        level_count: Number of pyramid levels.
        level_dimensions: Dimensions of each level in the pyramid.
        level_downsamples: Downsample factor for each level.
        mpp_x: Microns-per-pixel in the horizontal direction.
        mpp_y: Microns-per-pixel in the vertical direction.
        magnification: Nominal objective power (e.g. 20, 40) or None if unknown.
        associated_images: List of available associated image names (e.g. 'label').
        properties: Raw string-to-string metadata properties dictionary.
    """
    path: Path
    vendor: str
    dimensions: Tuple[int, int]
    level_count: int
    level_dimensions: List[Tuple[int, int]]
    level_downsamples: List[float]
    mpp_x: float
    mpp_y: float
    magnification: Optional[float]
    associated_images: List[str]
    properties: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "path": str(self.path),
            "vendor": self.vendor,
            "dimensions": self.dimensions,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "level_downsamples": self.level_downsamples,
            "mpp_x": self.mpp_x,
            "mpp_y": self.mpp_y,
            "magnification": self.magnification,
            "associated_images": self.associated_images,
            "properties": self.properties,
        }


def extract_metadata(reader: BaseWSI) -> SlideMetadata:
    """Extract and validate slide metadata from an opened WSI reader.

    Parameters
    ----------
    reader : BaseWSI
        An open slide reader instance.

    Returns
    -------
    SlideMetadata
        Standardized, validated metadata object.

    Raises
    ------
    MetadataExtractionError
        If slide dimensions or levels are corrupt, or if critical spatial scaling
        information (MPP) cannot be resolved.
    """
    if not reader.is_open:
        raise MetadataExtractionError(f"Cannot extract metadata. Slide is closed: {reader.path}")

    props = reader.properties
    path = reader.path

    # 1. Vendor detection
    vendor = props.get("openslide.vendor", "generic").lower()

    # 2. Dimensions & Level count validation
    try:
        level_count = reader.level_count
        level_dimensions = reader.level_dimensions
        level_downsamples = reader.level_downsamples
    except Exception as exc:
        raise MetadataExtractionError(
            f"Failed to retrieve pyramid properties from reader: {exc}"
        ) from exc

    if level_count <= 0:
        raise MetadataExtractionError(f"Slide has invalid level count: {level_count}")
    if len(level_dimensions) != level_count or len(level_downsamples) != level_count:
        raise MetadataExtractionError(
            f"Inconsistent level count vs. level arrays. count={level_count}, "
            f"dims={len(level_dimensions)}, downsamples={len(level_downsamples)}"
        )

    dimensions = level_dimensions[0]
    if dimensions[0] <= 0 or dimensions[1] <= 0:
        raise MetadataExtractionError(f"Slide has invalid full-res dimensions: {dimensions}")

    # 3. MPP extraction with fallbacks
    mpp_x = _parse_mpp(props, "openslide.mpp-x")
    mpp_y = _parse_mpp(props, "openslide.mpp-y")

    # If missing, try parsing from generic TIFF resolution tags
    if mpp_x is None or mpp_y is None:
        mpp_x, mpp_y = _extract_mpp_from_tiff_resolution(props)

    # If still missing, check description comments (common in Aperio/Leica)
    if mpp_x is None or mpp_y is None:
        mpp_x, mpp_y = _extract_mpp_from_description(props)

    # Final error if MPP cannot be resolved
    if mpp_x is None or mpp_y is None:
        raise MetadataExtractionError(
            f"Microns-per-pixel (MPP) could not be determined for slide {path}. "
            "Downstream patch sampling and model execution require physical pixel spacing."
        )

    # 4. Magnification (Objective power)
    magnification = _parse_float(props.get("openslide.objective-power"))
    if magnification is None:
        # Check vendor-specific fallbacks
        magnification = _parse_float(props.get("aperio.AppMag"))
    if magnification is None:
        magnification = _parse_float(props.get("leica.ObjectiveSenseMagnification"))
    if magnification is None:
        # Guess based on MPP (approximate check: 40x scan is ~0.25 MPP, 20x is ~0.50 MPP)
        if abs(mpp_x - 0.25) < 0.08:
            magnification = 40.0
        elif abs(mpp_x - 0.50) < 0.15:
            magnification = 20.0

    # 5. Associated images
    try:
        associated_images = reader.associated_images
    except Exception:
        associated_images = []

    # 6. Basic range verification
    if mpp_x <= 0.0 or mpp_y <= 0.0:
        raise MetadataExtractionError(f"Invalid negative or zero MPP values: ({mpp_x}, {mpp_y})")

    metadata = SlideMetadata(
        path=path,
        vendor=vendor,
        dimensions=dimensions,
        level_count=level_count,
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        magnification=magnification,
        associated_images=associated_images,
        properties=props,
    )

    logger.debug(
        "Extracted slide metadata",
        extra={
            "path": str(path),
            "vendor": vendor,
            "dimensions": f"{dimensions[0]}x{dimensions[1]}",
            "mpp": f"{mpp_x:.4f}x{mpp_y:.4f}",
            "magnification": magnification,
        },
    )

    return metadata


# ---------------------------------------------------------------------------
# Helper parsers
# ---------------------------------------------------------------------------

def _parse_float(val: str | None) -> float | None:
    """Parse a string to float safely, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_mpp(props: Dict[str, str], key: str) -> float | None:
    """Extract and parse float MPP value from properties."""
    return _parse_float(props.get(key))


def _extract_mpp_from_tiff_resolution(props: Dict[str, str]) -> Tuple[float | None, float | None]:
    """Calculate MPP from TIFF resolution tags.

    Resolution tags define pixels per unit. Converting to microns:
    - Unit 3 (centimeter) -> MPP = 10000 / Resolution
    - Unit 2 (inch)       -> MPP = 25400 / Resolution
    """
    unit = props.get("tiff.ResolutionUnit")
    x_res = _parse_float(props.get("tiff.XResolution"))
    y_res = _parse_float(props.get("tiff.YResolution"))

    if not x_res or not y_res:
        return None, None

    if unit == "centimeter":
        return 10000.0 / x_res, 10000.0 / y_res
    elif unit == "inch":
        return 25400.0 / x_res, 25400.0 / y_res

    return None, None


def _extract_mpp_from_description(props: Dict[str, str]) -> Tuple[float | None, float | None]:
    """Search for 'MPP = <float>' patterns inside TIFF comment tags."""
    comments = [
        props.get("openslide.comment", ""),
        props.get("tiff.ImageDescription", ""),
    ]

    pattern = re.compile(r"mpp\s*=\s*([0-9.]+)", re.IGNORECASE)

    for txt in comments:
        if not txt:
            continue
        match = pattern.search(txt)
        if match:
            try:
                mpp = float(match.group(1))
                return mpp, mpp
            except ValueError:
                pass

    return None, None
