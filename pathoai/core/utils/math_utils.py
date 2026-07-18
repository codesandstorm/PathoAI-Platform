"""
pathoai/core/utils/math_utils.py
=================================
Mathematical and coordinate utility functions for PathoAI-Platform.

Provides pure-Python/NumPy functions for:
- Coordinate system transformations (slide ↔ physical ↔ patch)
- Area computation (pixels → μm² → mm²)
- Geometric operations (centroid computation, IoU)

All functions are stateless, deterministic, and independently testable.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

from typing import List, Optional, Tuple

import numpy as np


def pixels_to_um2(pixel_count: int, mpp: float) -> float:
    """Convert pixel count (area) to square micrometers.

    Parameters
    ----------
    pixel_count : int
        Number of pixels representing the area.
    mpp : float
        Microns per pixel at the relevant pyramid level.

    Returns
    -------
    float
        Area in square micrometers (μm²).

    Examples
    --------
    >>> pixels_to_um2(1000, mpp=0.5)
    250.0
    """
    return float(pixel_count) * (mpp ** 2)


def pixels_to_mm2(pixel_count: int, mpp: float) -> float:
    """Convert pixel count (area) to square millimeters.

    Parameters
    ----------
    pixel_count : int
        Number of pixels representing the area.
    mpp : float
        Microns per pixel.

    Returns
    -------
    float
        Area in square millimeters (mm²).

    Examples
    --------
    >>> pixels_to_mm2(4_000_000, mpp=0.5)
    1.0
    """
    return pixels_to_um2(pixel_count, mpp) / 1e6


def um2_to_mm2(area_um2: float) -> float:
    """Convert area from μm² to mm².

    Parameters
    ----------
    area_um2 : float
        Area in square micrometers.

    Returns
    -------
    float
        Area in square millimeters.
    """
    return area_um2 / 1e6


def slide_to_level_coords(
    x_level0: int,
    y_level0: int,
    level_downsample: float,
) -> Tuple[int, int]:
    """Convert level-0 pixel coordinates to a given pyramid level.

    Parameters
    ----------
    x_level0 : int
        X coordinate at level 0.
    y_level0 : int
        Y coordinate at level 0.
    level_downsample : float
        Downsample factor for the target level (e.g., 4.0 for level 2 at 4× downsample).

    Returns
    -------
    Tuple[int, int]
        (x, y) coordinates at the target pyramid level.
    """
    return (
        int(x_level0 / level_downsample),
        int(y_level0 / level_downsample),
    )


def level_to_slide_coords(
    x_level: int,
    y_level: int,
    level_downsample: float,
) -> Tuple[int, int]:
    """Convert pyramid-level coordinates back to level-0 slide coordinates.

    Parameters
    ----------
    x_level : int
        X coordinate at the pyramid level.
    y_level : int
        Y coordinate at the pyramid level.
    level_downsample : float
        Downsample factor for the source level.

    Returns
    -------
    Tuple[int, int]
        (x, y) coordinates at level 0.
    """
    return (
        int(x_level * level_downsample),
        int(y_level * level_downsample),
    )


def compute_box_centroids(boxes: np.ndarray) -> np.ndarray:
    """Compute centroids (cx, cy) from bounding boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Bounding boxes in [x1, y1, x2, y2] format.
        Shape: (N, 4), float32 or int.

    Returns
    -------
    np.ndarray
        Centroid coordinates. Shape: (N, 2), float32. Columns: [cx, cy].

    Examples
    --------
    >>> import numpy as np
    >>> boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
    >>> compute_box_centroids(boxes)
    array([[ 5.,  5.],
           [10., 10.]], dtype=float32)
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes shape (N, 4), got {boxes.shape}")
    return ((boxes[:, :2] + boxes[:, 2:]) / 2.0).astype(np.float32)


def compute_box_areas(boxes: np.ndarray) -> np.ndarray:
    """Compute area of bounding boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Bounding boxes in [x1, y1, x2, y2] format. Shape: (N, 4).

    Returns
    -------
    np.ndarray
        Areas. Shape: (N,), float32.
    """
    widths = np.maximum(0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return (widths * heights).astype(np.float32)


def compute_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Intersection over Union (IoU) between two sets of boxes.

    Parameters
    ----------
    boxes_a : np.ndarray
        First set of boxes. Shape: (M, 4). Format: [x1, y1, x2, y2].
    boxes_b : np.ndarray
        Second set of boxes. Shape: (N, 4). Format: [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray
        IoU matrix. Shape: (M, N), float32.
        iou[i, j] is the IoU between boxes_a[i] and boxes_b[j].
    """
    # Expand for broadcasting: (M, 1, 4) and (1, N, 4)
    a = boxes_a[:, np.newaxis, :]  # (M, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, N, 4)

    # Intersection
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h  # (M, N)

    # Union
    area_a = compute_box_areas(boxes_a)[:, np.newaxis]  # (M, 1)
    area_b = compute_box_areas(boxes_b)[np.newaxis, :]  # (1, N)
    union_area = area_a + area_b - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def clip_boxes_to_image(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Clip bounding boxes to lie within image boundaries.

    Parameters
    ----------
    boxes : np.ndarray
        Bounding boxes. Shape: (N, 4). Format: [x1, y1, x2, y2].
    image_width : int
        Width of the image in pixels.
    image_height : int
        Height of the image in pixels.

    Returns
    -------
    np.ndarray
        Clipped boxes. Shape: (N, 4), same dtype as input.
    """
    clipped = boxes.copy().astype(np.float32)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, image_width)   # x1
    clipped[:, 1] = np.clip(clipped[:, 1], 0, image_height)  # y1
    clipped[:, 2] = np.clip(clipped[:, 2], 0, image_width)   # x2
    clipped[:, 3] = np.clip(clipped[:, 3], 0, image_height)  # y2
    return clipped


def find_best_pyramid_level(
    level_downsamples: List[float],
    target_mpp: float,
    slide_mpp: float,
) -> int:
    """Find the pyramid level whose resolution is closest to target_mpp.

    Parameters
    ----------
    level_downsamples : List[float]
        Downsample factors for each pyramid level (from WSIMetadata).
    target_mpp : float
        Target microns per pixel (e.g., 0.50 for 20× equivalent).
    slide_mpp : float
        MPP of the slide at level 0 (from WSIMetadata.mpp).

    Returns
    -------
    int
        Index of the best matching pyramid level.

    Examples
    --------
    >>> find_best_pyramid_level([1.0, 4.0, 16.0], target_mpp=0.50, slide_mpp=0.25)
    1  # Level 1 has MPP = 0.25 * 4.0 = 1.0 (closest to 0.50)
    """
    mpps = [slide_mpp * ds for ds in level_downsamples]
    differences = [abs(mpp - target_mpp) for mpp in mpps]
    return int(np.argmin(differences))


def bootstrap_confidence_interval(
    values: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    weights: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the weighted mean.

    Parameters
    ----------
    values : np.ndarray
        Array of values to bootstrap. Shape: (N,).
    n_resamples : int
        Number of bootstrap resamples. Default: 1000.
    confidence : float
        Confidence level. Default: 0.95 (gives 95% CI).
    weights : np.ndarray, optional
        Weights for weighted mean. If None, uses uniform weights.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) of the confidence interval.

    Examples
    --------
    >>> import numpy as np
    >>> vals = np.array([10.0, 20.0, 15.0, 25.0, 18.0])
    >>> ci_lower, ci_upper = bootstrap_confidence_interval(vals, n_resamples=500)
    >>> print(f"95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")
    """
    rng = np.random.RandomState(seed)
    n = len(values)

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = weights / weights.sum()

    bootstrap_means = np.empty(n_resamples, dtype=np.float64)
    indices = np.arange(n)

    for i in range(n_resamples):
        sample_idx = rng.choice(indices, size=n, replace=True)
        sample_vals = values[sample_idx]
        sample_weights = weights[sample_idx]
        # Renormalize weights for the sample
        sample_weight_sum = sample_weights.sum()
        if sample_weight_sum > 0:
            bootstrap_means[i] = np.sum(sample_vals * sample_weights / sample_weight_sum)
        else:
            bootstrap_means[i] = np.mean(sample_vals)

    alpha = 1.0 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(lower), float(upper)
