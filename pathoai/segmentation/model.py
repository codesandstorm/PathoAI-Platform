"""
pathoai/segmentation/model.py
=============================
Standardized Model Wrapper for Semantic Segmentation.

Wraps the core neural network module and provides unified interfaces
for weights serialization, parameter counting, device management,
and encoder parameter freezing/unfreezing.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.4
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class SegmentationModel(nn.Module):
    """Unified wrapper around semantic segmentation architectures."""

    def __init__(self, model: nn.Module) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The raw neural network architecture class.
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch input images of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, C, H, W).
        """
        return self.model(x)

    def get_encoder(self) -> Optional[nn.Module]:
        """Resolve and return the model backbone/encoder module if present."""
        # 1. Directly on wrapped model
        if hasattr(self.model, "encoder"):
            return self.model.encoder

        # 2. Nested inside custom architecture wraps (like DeepLabV3PlusArch)
        if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            return self.model.model.encoder

        return None

    def freeze_encoder(self) -> None:
        """Freeze all parameters in the encoder backbone."""
        encoder = self.get_encoder()
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad = False
            logger.info("Froze encoder backbone parameters.")
        else:
            logger.warning("freeze_encoder: No encoder module found. Skipping.")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all parameters in the encoder backbone."""
        encoder = self.get_encoder()
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad = True
            logger.info("Unfroze encoder backbone parameters.")
        else:
            logger.warning("unfreeze_encoder: No encoder module found. Skipping.")

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters of the model.

        Returns
        -------
        Dict[str, int]
            Dictionary containing 'total', 'trainable', and 'non_trainable' counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }

    def save_weights(self, path: str | Path) -> None:
        """Serialize model weights state_dict to a file.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)
        logger.info("Model weights saved to %s", p)

    def load_weights(self, path: str | Path) -> None:
        """Load state_dict weights from a file.

        Parameters
        ----------
        path : str | Path
            Source weights checkpoint file path.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Weights checkpoint file not found: {p}")

        state_dict = torch.load(p, map_location="cpu")

        # If the state_dict was saved from a Trainer payload, extract weights
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        self.load_state_dict(state_dict)
        logger.info("Model weights loaded from %s", p)

    def to_device(self, device: Union[str, torch.device]) -> SegmentationModel:
        """Route the wrapped model parameters to target hardware device."""
        self.to(device)
        logger.debug("Routed SegmentationModel parameters to %s", device)
        return self
