"""
pathoai/detection/factory.py
============================
Model Factory for object detection architectures.

Reads the detection configuration object, resolves registered detector classes,
and instantiates concrete model instances.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.3
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from pathoai.core.logger import get_logger
from pathoai.detection.registry import get_detector_class

# Import architectures package to trigger registry decoration
import pathoai.detection.architectures  # noqa: F401

logger = get_logger(__name__)


def create_detector(config: Any) -> nn.Module:
    """Create a object detection model instance from the configuration object.

    Parameters
    ----------
    config : Any
        Global configuration (ConfigNode instance or dict).

    Returns
    -------
    nn.Module
        Instantiated PyTorch object detection model.
    """
    if hasattr(config, "detection"):
        det_cfg = config.detection
    else:
        det_cfg = config

    arch_key = getattr(det_cfg, "architecture", "yolo")
    n_classes = getattr(det_cfg, "n_classes", 4)
    in_channels = getattr(det_cfg, "in_channels", 3)

    logger.info(
        "Creating detection model",
        extra={
            "architecture": arch_key,
            "n_classes": n_classes,
            "in_channels": in_channels,
        },
    )

    # Fetch architecture class from registry
    detector_cls = get_detector_class(arch_key)

    # Instantiate model
    model = detector_cls(
        in_channels=in_channels,
        n_classes=n_classes,
    )

    return model
