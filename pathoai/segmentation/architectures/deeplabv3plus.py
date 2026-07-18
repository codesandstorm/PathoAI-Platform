"""
pathoai/segmentation/architectures/deeplabv3plus.py
===================================================
DeepLabV3+ Semantic Segmentation Architecture wrapper.

Wraps the Segmentation Models PyTorch (SMP) implementation of DeepLabV3+,
configuring encoders, pre-trained weights, and target classes.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.3
"""

from __future__ import annotations

from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from pathoai.segmentation.registry import register_model


@register_model("deeplabv3plus")
class DeepLabV3PlusArch(nn.Module):
    """DeepLabV3+ semantic segmentation model wrapper for SMP integration."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        n_classes: int = 6,
        activation: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        encoder_name : str
            Backbone name (e.g. 'resnet34', 'efficientnet-b3').
        encoder_weights : str, optional
            Pretrained weights name (e.g. 'imagenet' or None).
        n_classes : int
            Number of output segmentation classes.
        activation : str, optional
            Output activation function (e.g. 'softmax2d', 'sigmoid', or None).
        """
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=n_classes,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DeepLabV3+.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, C, H, W).
        """
        return self.model(x)
