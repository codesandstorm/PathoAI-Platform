"""
pathoai/segmentation/losses.py
=============================
Loss Factory and Custom Loss functions for semantic segmentation.

Provides:
    DiceLoss: Multiclass Dice loss.
    FocalLoss: Multiclass Focal loss.
    CombinedLoss: Combines weighted loss components (e.g. CrossEntropy + Dice).
    LossFactory: Instantiates the loss function based on config parameters.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.5
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class DiceLoss(nn.Module):
    """Multiclass Dice Loss with channel-wise averaging."""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Unnormalized logits of shape (B, C, H, W).
        targets : torch.Tensor
            Target class IDs of shape (B, H, W).
        """
        n_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encoding targets: shape (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=n_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        # Average Dice loss across classes (excluding background class 0 is common,
        # but standard is averaging over all classes)
        for c in range(n_classes):
            p_c = probs[:, c, ...]
            t_c = targets_one_hot[:, c, ...]

            intersection = torch.sum(p_c * t_c)
            cardinality = torch.sum(p_c + t_c)

            dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
            dice_loss += (1.0 - dice_c)

        return dice_loss / n_classes


class FocalLoss(nn.Module):
    """Multiclass Focal Loss."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (B, C, H, W).
        targets : torch.Tensor
            Class IDs of shape (B, H, W).
        """
        log_probs = F.log_softmax(logits, dim=1)
        # NLL loss returns negative log probabilities for target classes
        nll = F.nll_loss(log_probs, targets, reduction="none")

        # Exponentiate to get probability of correct class (p_t)
        probs = torch.exp(-nll)

        # Compute focal scaling factor
        focal_weight = (1.0 - probs) ** self.gamma
        loss = focal_weight * nll

        if self.alpha is not None:
            # Gather alpha values for target classes
            # targets is shape (B, H, W) -> flatten to index alpha
            alpha_class = self.alpha[targets]
            loss = alpha_class * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """Combines multiple loss components with scale weights (e.g. 1.0 * CE + 1.0 * Dice)."""

    def __init__(self, losses: List[Tuple[float, nn.Module]]) -> None:
        """
        Parameters
        ----------
        losses : List[Tuple[float, nn.Module]]
            List of (weight, loss_module) pairs.
        """
        super().__init__()
        self.losses = nn.ModuleList([loss[1] for loss in losses])
        self.weights = [loss[0] for loss in losses]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for weight, loss_fn in zip(self.weights, self.losses):
            total_loss += weight * loss_fn(logits, targets)
        return total_loss


class LossFactory:
    """Factory creating configured loss functions for segmentation."""

    @staticmethod
    def create_loss(config: Any, class_weights: Optional[List[float]] = None) -> nn.Module:
        """Create the loss module based on configuration options.

        Parameters
        ----------
        config : Any
            Global configuration (ConfigNode instance).
        class_weights : List[float], optional
            Weights vector for class-imbalanced losses (e.g., CrossEntropy / Focal).

        Returns
        -------
        nn.Module
            Instantiated loss function.
        """
        seg_cfg = config.segmentation
        loss_name = seg_cfg.get("loss_name", "ce").lower()

        # Convert class weights list to torch Tensor
        weights_tensor: Optional[torch.Tensor] = None
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        logger.info("Initializing segmentation loss function", extra={"loss_name": loss_name, "has_weights": class_weights is not None})

        if loss_name == "ce":
            return nn.CrossEntropyLoss(weight=weights_tensor)

        elif loss_name == "dice":
            return DiceLoss()

        elif loss_name == "focal":
            return FocalLoss(alpha=weights_tensor)

        elif loss_name == "lovasz":
            # smp.losses.LovaszLoss requires mode="multiclass"
            return smp.losses.LovaszLoss(mode="multiclass")

        elif loss_name == "dice_ce" or loss_name == "combined":
            # 1.0 * CrossEntropy + 1.0 * Dice
            ce = nn.CrossEntropyLoss(weight=weights_tensor)
            dice = DiceLoss()
            return CombinedLoss([(1.0, ce), (1.0, dice)])

        else:
            raise ValidationError(
                f"Unsupported loss name: '{loss_name}'. "
                f"Supported values: 'ce', 'dice', 'focal', 'lovasz', 'dice_ce', 'combined'"
            )
