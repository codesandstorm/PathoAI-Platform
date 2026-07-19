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

    # Resolve encoder and architecture key from composite name if needed
    if "deeplabv3plus" in model_name.lower():
        arch_key = "deeplabv3plus"
        parts = model_name.split("_", 1)
        if len(parts) > 1:
            enc_part = parts[1].replace("_", "-")
            if "resnet" in enc_part:
                enc_part = enc_part.replace("-", "")
            encoder_name = enc_part
        else:
            encoder_name = seg_cfg.get("encoder_name", "resnet34")
    else:
        arch_key = model_name
        encoder_name = seg_cfg.get("encoder_name", "resnet34")

    # Pretrained weights defaults
    encoder_weights = seg_cfg.get("encoder_weights", "imagenet")

    logger.info(
        "Creating segmentation model",
        extra={
            "model_name": model_name,
            "arch_key": arch_key,
            "n_classes": n_classes,
            "encoder": encoder_name,
            "pretrained": encoder_weights,
        },
    )

    # Fetch the architecture class from registry
    arch_class = get_model_class(arch_key)

    # Instantiate model
    model = arch_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        n_classes=n_classes,
    )

    return model
