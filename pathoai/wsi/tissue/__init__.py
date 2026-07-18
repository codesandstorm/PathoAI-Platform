"""pathoai.wsi.tissue — Classical tissue detection and background removal.

Exposes:
    TissueDetector: Class to generate binary tissue masks from slide thumbnails.
"""

from pathoai.wsi.tissue.tissue import TissueDetector

__all__ = [
    "TissueDetector",
]
