"""pathoai.segmentation — Semantic Segmentation Engine package.

Exposes:
    create_model: Factory to instantiate models from configuration.
    SegmentationModel: Standardized model wrapper class.
    LossFactory: Loss function factory.
    estimate_model_size_mb: Estimates memory usage size in MB.
    verify_output_shape: Validates input/output shape mapping.
    check_backbone: Checks if backbone is supported.
    generate_model_summary: Writes model parameter summaries to disk.
    SegmentationInference: Runs prediction overlays and batch maps.
    export_model: Compiles models to ONNX/TorchScript.
    register_model: Decorator to register architectures.
    get_model_class: Resolves registered classes.
    list_registered_models: Lists registered model names.
"""

from pathoai.segmentation.export import export_model
from pathoai.segmentation.factory import create_model
from pathoai.segmentation.inference import SegmentationInference
from pathoai.segmentation.losses import LossFactory
from pathoai.segmentation.model import SegmentationModel
from pathoai.segmentation.registry import (
    get_model_class,
    list_registered_models,
    register_model,
)
from pathoai.segmentation.summary import generate_model_summary
from pathoai.segmentation.utils import (
    check_backbone,
    estimate_model_size_mb,
    verify_output_shape,
)

__all__ = [
    "create_model",
    "SegmentationModel",
    "LossFactory",
    "estimate_model_size_mb",
    "verify_output_shape",
    "check_backbone",
    "generate_model_summary",
    "SegmentationInference",
    "export_model",
    "register_model",
    "get_model_class",
    "list_registered_models",
]
