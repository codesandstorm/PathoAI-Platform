"""
pathoai/segmentation/utils.py
=============================
Utility functions for semantic segmentation models.

Includes:
    estimate_model_size: Computes model size in MB (float).
    verify_output_shape: Validates output shapes for target input dimensions.
    check_backbone: Checks if a backbone is supported by SMP.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.6
"""

from __future__ import annotations

from typing import Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate the memory size of a model's parameters and buffers in Megabytes.

    Parameters
    ----------
    model : nn.Module
        The PyTorch module.

    Returns
    -------
    float
        Size in MB.
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024**2
    return float(size_all_mb)


def verify_output_shape(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
) -> Tuple[int, int, int, int]:
    """Pass a dummy batch through the model to verify output dimensions.

    Parameters
    ----------
    model : nn.Module
        The model to verify.
    input_shape : Tuple[int, int, int, int]
        Shape of dummy tensor (B, C, H, W).

    Returns
    -------
    Tuple[int, int, int, int]
        Shape of output tensor.
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_shape, dtype=torch.float32, device=device)

    with torch.no_grad():
        dummy_output = model(dummy_input)

    return tuple(dummy_output.shape)


def check_backbone(backbone_name: str) -> bool:
    """Check if the given backbone name is supported by SMP encoders.

    Parameters
    ----------
    backbone_name : str
        Encoder backbone identifier (e.g. 'resnet34').

    Returns
    -------
    bool
        True if supported, False otherwise.
    """
    from segmentation_models_pytorch.encoders import encoders
    supported = backbone_name.lower() in encoders
    logger.debug("Checking backbone availability: %s -> %s", backbone_name, supported)
    return supported
