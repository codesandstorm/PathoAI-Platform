"""
pathoai/segmentation/factory.py
==============================
Model Factory for semantic segmentation architectures.

Reads the pipeline configuration object, resolves registered model classes,
and instantiates the concrete model with correct parameters.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.2
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from pathoai.core.logger import get_logger
from pathoai.segmentation.registry import get_model_class

# Import architectures package to trigger registry decoration
import pathoai.segmentation.architectures  # noqa: F401

logger = get_logger(__name__)


def create_model(config: Any) -> nn.Module:
    """Create a segmentation model instance from the configuration object.

    Reads model name, backbone/encoder, pre-trained weights, and target classes
    parameters from the 'segmentation' config section.

    Parameters
    ----------
    config : Any
        Global configuration (ConfigNode instance).

    Returns
    -------
    nn.Module
        Instantiated PyTorch neural network model.
    """
    seg_cfg = config.segmentation
    model_name = seg_cfg.model_name
    n_classes = seg_cfg.n_classes

    # Fetch custom properties from configuration sections if present
    # We resolve encoder parameter names which can differ across backbones
    encoder_name = getattr(seg_cfg, "encoder_name", "resnet34")
    encoder_weights = getattr(seg_cfg, "encoder_weights", "imagenet")

    logger.info(
        "Creating segmentation model",
        extra={
            "model_name": model_name,
            "n_classes": n_classes,
            "encoder": encoder_name,
            "pretrained": encoder_weights,
        },
    )

    # Fetch the architecture class from registry
    arch_class = get_model_class(model_name)

    # Instantiate model
    model = arch_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        n_classes=n_classes,
    )

    return model
