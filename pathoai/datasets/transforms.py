"""
pathoai/datasets/transforms.py
==============================
Image augmentations and normalization transforms using Albumentations.

Defines training augmentations (rotations, flips, brightness/contrast jitter, noise)
and deterministic validation/testing normalizations using ImageNet statistics.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathoai.core.constants import IMAGENET_MEAN, IMAGENET_STD
from pathoai.core.logger import get_logger

logger = get_logger(__name__)


def get_transforms(
    split: str,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    brightness_contrast_prob: float = 0.5,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    gaussian_noise_prob: float = 0.2,
) -> A.Compose:
    """Retrieve Albumentations Compose pipeline for image and mask transforms.

    Parameters
    ----------
    split : str
        The data split ('train', 'val', or 'test').
    flip_prob : float
        Probability of random horizontal and vertical flips.
    rotate_prob : float
        Probability of random 90-degree rotations.
    brightness_contrast_prob : float
        Probability of random brightness/contrast jitter.
    brightness_limit : float
        Jitter limit for brightness.
    contrast_limit : float
        Jitter limit for contrast.
    gaussian_noise_prob : float
        Probability of adding Gaussian noise.

    Returns
    -------
    A.Compose
        Albumentations composition of transforms.
    """
    # Base normalization is applied to all splits
    normalization = A.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        max_pixel_value=255.0,
    )
    to_tensor = ToTensorV2()

    if split == "train":
        logger.debug("Building training augmentations Compose pipeline")
        transforms_list = [
            A.HorizontalFlip(p=flip_prob),
            A.VerticalFlip(p=flip_prob),
            A.RandomRotate90(p=rotate_prob),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=brightness_contrast_prob,
            ),
            A.GaussNoise(p=gaussian_noise_prob),
            normalization,
            to_tensor,
        ]
    else:
        logger.debug("Building deterministic validation/testing Compose pipeline")
        transforms_list = [
            normalization,
            to_tensor,
        ]

    return A.Compose(transforms_list)
