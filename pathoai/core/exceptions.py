"""
pathoai/core/exceptions.py
==========================
Custom exception hierarchy for PathoAI-Platform.

All exceptions inherit from PathoAIException, enabling callers to
catch either specific or broad PathoAI errors as appropriate.

Design principles:
- Every exception includes a descriptive, actionable message
- Exceptions preserve the original cause via `raise X from original`
- No bare `except Exception` in production code — use specific types
- Exception names describe WHAT went wrong, not WHERE

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""


class PathoAIException(Exception):
    """Base exception for all PathoAI-Platform errors.

    All custom exceptions in this codebase inherit from this class,
    enabling callers to catch all PathoAI-specific errors with a
    single `except PathoAIException` if needed.
    """


# ===========================================================================
# CONFIGURATION ERRORS
# ===========================================================================

class ConfigurationError(PathoAIException):
    """Raised when a configuration file is invalid, missing, or has schema errors.

    Examples:
        - Required config key is absent
        - Config value is outside allowed range
        - Config file is not valid YAML
        - Incompatible config combination (e.g., CUDA requested but not available)
    """


# ===========================================================================
# ENVIRONMENT ERRORS
# ===========================================================================

class EnvironmentValidationError(PathoAIException):
    """Raised when the runtime environment does not meet requirements.

    Examples:
        - Wrong Python version
        - Required package not installed
        - Insufficient RAM or disk space
        - OpenSlide binaries not found
    """


# ===========================================================================
# DATA ERRORS
# ===========================================================================

class DataError(PathoAIException):
    """Base class for all data-related errors."""


class WSIReadError(DataError):
    """Raised when a Whole Slide Image cannot be opened or read.

    Examples:
        - File does not exist
        - File format not supported by OpenSlide
        - File is corrupted or truncated
        - OpenSlide library not installed or binaries missing
        - Insufficient permissions to read the file
    """


class MetadataExtractionError(DataError):
    """Raised when required metadata cannot be extracted from a WSI.

    Examples:
        - MPP (microns per pixel) not available in slide properties
        - Objective power property missing
        - Inconsistent level dimensions
    """


class PatchExtractionError(DataError):
    """Raised when patch extraction from a WSI fails.

    Examples:
        - Requested coordinates are outside slide bounds
        - Memory allocation failure during region read
        - Corrupt tile data at requested location
    """


class DatasetValidationError(DataError):
    """Raised when a dataset fails integrity validation.

    Examples:
        - Expected slides are missing from disk
        - Annotation mask dimensions don't match slide dimensions
        - sTIL score CSV is missing entries
        - Class labels in annotation mask are outside valid range
    """


class StainNormalizationError(DataError):
    """Raised when stain normalization fails for a patch.

    Examples:
        - SVD decomposition fails on degenerate tissue
        - Stain matrix is singular
        - Reference stain matrix has invalid shape
    """


# ===========================================================================
# MODEL ERRORS
# ===========================================================================

class ModelError(PathoAIException):
    """Base class for all model-related errors."""


class CheckpointLoadError(ModelError):
    """Raised when a model checkpoint cannot be loaded.

    Examples:
        - Checkpoint file does not exist
        - Checkpoint architecture doesn't match current model definition
        - Missing expected state dict keys
        - SHA-256 hash mismatch (corrupted checkpoint)
    """


class InferenceError(ModelError):
    """Raised when model inference fails.

    Examples:
        - CUDA out of memory (OOM) error
        - NaN outputs detected in model predictions
        - Input tensor shape mismatch with model expectations
    """


class ModelArchitectureError(ModelError):
    """Raised when a model architecture is incorrectly configured.

    Examples:
        - Model name not found in registry
        - Incompatible number of classes vs. checkpoint
        - Invalid hyperparameter combination
    """


# ===========================================================================
# FUSION ERRORS
# ===========================================================================

class FusionError(PathoAIException):
    """Raised when the spatial fusion stage fails.

    Examples:
        - Segmentation mask and detection result have mismatched coordinate systems
        - sTIL score computation fails due to zero stroma area
        - Patch coordinate map doesn't cover the full slide
    """


class sTILComputationError(FusionError):
    """Raised when the sTIL score cannot be computed.

    Examples:
        - Stroma area is zero (division by zero prevented)
        - No tissue patches available for scoring
        - Aggregation weights sum to zero
    """


# ===========================================================================
# VALIDATION ERRORS
# ===========================================================================

class ValidationError(PathoAIException):
    """Raised when statistical validation of results fails.

    Examples:
        - Bootstrap CI estimation fails (degenerate distribution)
        - Required ground truth labels are missing for evaluation
        - Score distribution fails quality checks
    """


# ===========================================================================
# REPORT ERRORS
# ===========================================================================

class ReportGenerationError(PathoAIException):
    """Raised when report generation fails.

    Examples:
        - Output directory is not writable
        - JSON serialization fails for non-serializable type
        - PDF template rendering fails
    """


# ===========================================================================
# PIPELINE ERRORS
# ===========================================================================

class PipelineError(PathoAIException):
    """Raised for pipeline orchestration errors.

    Examples:
        - Stage dependencies not satisfied
        - Resume checkpoint is incompatible with current config
        - Unexpected stage completion status
    """
