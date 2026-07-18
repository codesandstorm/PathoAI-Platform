"""
pathoai/core/__init__.py
========================
Platform Core package.
"""
from pathoai.core.config import ConfigManager, get_config, load_config
from pathoai.core.constants import (
    CELL_CLASSES,
    LYMPHOCYTE_DET_CLASS_ID,
    N_TISSUE_CLASSES,
    STROMA_CLASS_ID,
    TISSUE_CLASSES,
)
from pathoai.core.exceptions import (
    ConfigurationError,
    DataError,
    FusionError,
    ModelError,
    PathoAIException,
    ValidationError,
    WSIReadError,
)
from pathoai.core.logger import configure_logging, get_logger
from pathoai.core.reproducibility import set_global_seed
from pathoai.core.types import (
    CellDetectionResult,
    PatchCoordinateMap,
    PatchSTILScore,
    SegmentationResult,
    ValidationReport,
    WSIMetadata,
    sTILResult,
)

__all__ = [
    # Config
    "ConfigManager", "get_config", "load_config",
    # Constants
    "CELL_CLASSES", "TISSUE_CLASSES", "STROMA_CLASS_ID",
    "LYMPHOCYTE_DET_CLASS_ID", "N_TISSUE_CLASSES",
    # Exceptions
    "PathoAIException", "ConfigurationError", "DataError",
    "WSIReadError", "ModelError", "FusionError", "ValidationError",
    # Logging
    "configure_logging", "get_logger",
    # Reproducibility
    "set_global_seed",
    # Types
    "WSIMetadata", "PatchCoordinateMap", "SegmentationResult",
    "CellDetectionResult", "PatchSTILScore", "sTILResult", "ValidationReport",
]
