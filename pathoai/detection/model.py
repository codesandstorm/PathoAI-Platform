"""
pathoai/detection/model.py
==========================
DetectionModel PyTorch Wrapper.

Provides device routing, checkpoint save/load, parameter freezing/unfreezing,
and state management for object detection models.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.4
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class DetectionModel:
    """Wrapper around PyTorch neural network detection architectures."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            Instantiated PyTorch detector architecture.
        device : Optional[torch.device]
            Target hardware device (CPU or CUDA). Defaults to CPU if unspecified.
        """
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def to(self, device: Union[str, torch.device]) -> "DetectionModel":
        """Move the model to the target device.

        Parameters
        ----------
        device : str or torch.device
            Target hardware device.

        Returns
        -------
        DetectionModel
            Self reference.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model.to(self.device)
        return self

    def eval(self) -> "DetectionModel":
        """Set the model to evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "DetectionModel":
        """Set the model to training mode."""
        self.model.train(mode)
        return self

    def freeze_backbone(self) -> None:
        """Freeze stem and backbone parameters for transfer learning."""
        for name, param in self.model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        logger.info("Frozen detector backbone parameters")

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all detector parameters")

    def save_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model state dict and optional metadata to checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)
        logger.info("Saved detector checkpoint: %s", path)

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load model state dict and metadata from checkpoint file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        logger.info("Loaded detector checkpoint: %s", path)
        return checkpoint.get("metadata", {})

    def predict_patch(
        self, patch_tensor: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Run forward inference on a single patch or batch tensor of patches.

        Parameters
        ----------
        patch_tensor : torch.Tensor
            Image tensor of shape (B, C, H, W) or (C, H, W).

        Returns
        -------
        List[Dict[str, torch.Tensor]]
            Decoded bounding boxes, scores, and labels per image.
        """
        if patch_tensor.ndim == 3:
            patch_tensor = patch_tensor.unsqueeze(0)

        patch_tensor = patch_tensor.to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(patch_tensor)

        return outputs
