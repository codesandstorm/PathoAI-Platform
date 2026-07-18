"""
pathoai/datasets/statistics.py
================================
Dataset profiling, statistics, and class weight calculator.

Aggregates pixel-level class counts from manifest entries, computes class imbalances,
and calculates class weights (inverse frequency, smooth inverse frequency) for loss functions.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


def compute_class_frequencies(
    manifest_entries: List[Dict[str, Any]],
    n_classes: int = 6,
) -> Dict[int, int]:
    """Aggregate pixel counts per class from manifest entries.

    Parameters
    ----------
    manifest_entries : List[Dict[str, Any]]
        List of manifest entries.
    n_classes : int
        Number of classes. Defaults to 6 (TIGER schema).

    Returns
    -------
    Dict[int, int]
        Dictionary mapping class ID (int) -> total pixel count.
    """
    totals = {c: 0 for c in range(n_classes)}

    for entry in manifest_entries:
        dist = entry.get("class_distribution", {})
        for class_str, count in dist.items():
            try:
                class_id = int(class_str)
                if class_id in totals:
                    totals[class_id] += int(count)
            except ValueError:
                pass

    return totals


def calculate_class_loss_weights(
    class_counts: Dict[int, int],
    method: str = "inverse_frequency",
    n_classes: int = 6,
) -> List[float]:
    """Calculate class weights for loss functions to combat class imbalance.

    Supports 'inverse_frequency' and 'median_frequency' / 'smooth_inverse_frequency'.

    Parameters
    ----------
    class_counts : Dict[int, int]
        Total pixel count per class ID.
    method : str
        Method: 'inverse_frequency' or 'smooth_inverse_frequency'.
    n_classes : int
        Number of target classes.

    Returns
    -------
    List[float]
        List of floats containing the computed weight for each class.
    """
    counts = np.array([class_counts.get(c, 0) for c in range(n_classes)], dtype=np.float64)
    total_pixels = np.sum(counts)

    if total_pixels == 0:
        # Fallback to uniform weights if no pixels recorded
        return [1.0] * n_classes

    weights = np.ones(n_classes, dtype=np.float64)

    if method == "inverse_frequency":
        # w_c = total / (C * count_c)
        for c in range(n_classes):
            c_count = counts[c]
            if c_count > 0:
                weights[c] = total_pixels / (n_classes * c_count)
            else:
                weights[c] = 0.0  # Zero weight for absent classes

    elif method == "smooth_inverse_frequency":
        # w_c = 1 / log(1.2 + count_c / total)
        for c in range(n_classes):
            freq = counts[c] / total_pixels
            weights[c] = 1.0 / math.log(1.2 + freq)
    else:
        logger.warning("Unknown weights method: %s. Using uniform weights.", method)

    # Normalize weights so that the mean is 1.0 (stabilizes gradients)
    mean_w = np.mean(weights[weights > 0]) if np.any(weights > 0) else 1.0
    if mean_w > 0:
        weights = weights / mean_w

    return weights.tolist()


def generate_dataset_statistics_report(
    train_entries: List[Dict[str, Any]],
    val_entries: List[Dict[str, Any]],
    test_entries: List[Dict[str, Any]],
    n_classes: int = 6,
) -> Dict[str, Any]:
    """Compile a comprehensive statistical profile of the dataset splits.

    Parameters
    ----------
    train_entries : List[Dict[str, Any]]
        List of training manifest entries.
    val_entries : List[Dict[str, Any]]
        List of validation manifest entries.
    test_entries : List[Dict[str, Any]]
        List of test manifest entries.
    n_classes : int
        Number of target classes.

    Returns
    -------
    Dict[str, Any]
        Dictionary report of splits, patches, and class frequencies.
    """
    splits = {"train": train_entries, "val": val_entries, "test": test_entries}

    report: Dict[str, Any] = {
        "summary": {
            "total_patches": sum(len(lst) for lst in splits.values()),
            "split_counts": {name: len(lst) for name, lst in splits.items()},
        },
        "splits_profile": {},
    }

    for name, entries in splits.items():
        if not entries:
            continue

        class_counts = compute_class_frequencies(entries, n_classes)
        total_p_pixels = sum(class_counts.values())

        # Average tissue coverage
        avg_coverage = float(np.mean([e.get("tissue_coverage", 0.0) for e in entries]))

        report["splits_profile"][name] = {
            "n_patches": len(entries),
            "avg_tissue_coverage": round(avg_coverage, 4),
            "class_frequencies": {
                str(c): {
                    "pixel_count": class_counts[c],
                    "pixel_fraction": round(class_counts[c] / total_p_pixels, 6)
                    if total_p_pixels > 0
                    else 0.0,
                }
                for c in range(n_classes)
            },
        }

    return report
