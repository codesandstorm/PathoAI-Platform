"""
pathoai/validation/utils.py
===========================
Validation Math & Statistical Helpers.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.19
"""

from __future__ import annotations

import numpy as np


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))
