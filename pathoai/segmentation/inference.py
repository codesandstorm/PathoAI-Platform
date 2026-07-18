"""
pathoai/segmentation/inference.py
=================================
Inference Engine for semantic segmentation models.

Includes batch predictions, softmax/argmax conversions, and overlays generation.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.8
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pathoai.core.constants import IMAGENET_MEAN, IMAGENET_STD
from pathoai.core.logger import get_logger
from pathoai.visualization.colormap import colorize_mask

logger = get_logger(__name__)


def preprocess_numpy_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess a raw numpy image array into a normalized tensor.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3) in range [0, 255].

    Returns
    -------
    torch.Tensor
        Normalized channel-first tensor of shape (1, 3, H, W).
    """
    img = image.astype(np.float32) / 255.0
    # Apply ImageNet normalization
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std

    # Transpose to (3, H, W)
    img_t = np.transpose(img, (2, 0, 1))
    # Add batch dimension -> (1, 3, H, W)
    tensor = torch.from_numpy(img_t).unsqueeze(0)
    return tensor


class SegmentationInference:
    """Handles running model inference over patches, batches, and slides."""

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The trained model (either raw or wrapped SegmentationModel).
        device : str
            Execution device.
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    def predict_patch(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict segmentation class IDs for a single patch image.

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            RGB image patch. If numpy, shape is (H, W, 3). If tensor, shape is (3, H, W)
            or (1, 3, H, W).

        Returns
        -------
        np.ndarray
            Class IDs array of shape (H, W).
        """
        if isinstance(image, np.ndarray):
            x = preprocess_numpy_image(image).to(self.device)
        else:
            x = image.to(self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            # Perform argmax along class channel
            preds = logits.argmax(dim=1).squeeze(0)

        return preds.detach().cpu().numpy().astype(np.uint8)

    def predict_batch(self, images_batch: torch.Tensor) -> torch.Tensor:
        """Predict segmentation class IDs for a batch of images.

        Parameters
        ----------
        images_batch : torch.Tensor
            Batch tensor of shape (B, 3, H, W) normalized.

        Returns
        -------
        torch.Tensor
            Class IDs tensor of shape (B, H, W) on CPU.
        """
        x = images_batch.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            preds = logits.argmax(dim=1)

        return preds.detach().cpu()

    def predict_probabilities(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict softmax probability maps for a patch.

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            RGB patch.

        Returns
        -------
        np.ndarray
            Softmax probability map of shape (C, H, W).
        """
        if isinstance(image, np.ndarray):
            x = preprocess_numpy_image(image).to(self.device)
        else:
            x = image.to(self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        return probs.detach().cpu().numpy()

    def save_prediction_mask(self, mask: np.ndarray, output_path: str | Path) -> None:
        """Save a predicted class ID mask to disk as an 8-bit PNG.

        Parameters
        ----------
        mask : np.ndarray
            Class IDs mask array (H, W).
        output_path : str | Path
            Destination path.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(mask.astype(np.uint8), mode="L")
        img.save(out)
        logger.debug("Saved prediction mask to %s", out)

    def save_prediction_overlay(
        self,
        mask: np.ndarray,
        output_path: str | Path,
        alpha: int = 150,
    ) -> None:
        """Save a colorized predicted mask overlay using the TIGER colormap.

        Parameters
        ----------
        mask : np.ndarray
            Class IDs mask array (H, W).
        output_path : str | Path
            Destination path.
        alpha : int
            Overlay transparency (0 to 255).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Colorize using colormap utilities
        colored_rgba = colorize_mask(mask, alpha=alpha)
        img = Image.fromarray(colored_rgba, mode="RGBA")
        img.save(out)
        logger.debug("Saved prediction overlay color image to %s", out)
