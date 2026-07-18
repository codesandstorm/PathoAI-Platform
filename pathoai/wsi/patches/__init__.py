"""pathoai.wsi.patches — Uniform tissue-aware WSI patch sampling.

Exposes:
    PatchMetadata: Dataclass storing patch coordinates and metadata.
    PatchSampler: Class to sample grid patches from slides based on tissue coverage.
"""

from pathoai.wsi.patches.patches import PatchMetadata, PatchSampler

__all__ = [
    "PatchMetadata",
    "PatchSampler",
]
