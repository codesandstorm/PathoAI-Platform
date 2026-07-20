"""
pathoai/validation/calibration.py
=================================
Model Calibration Engine.

Computes Expected Calibration Error (ECE) for model confidence scores.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.10
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class CalibrationEngine:
    """Computes calibration error metrics."""

    def compute_ece(
        self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        """Computes Expected Calibration Error (ECE)."""
        confs = np.asarray(confidences, dtype=np.float64)
        accs = np.asarray(accuracies, dtype=np.float64)

        if len(confs) == 0:
            return {"ece": 0.0, "max_calibration_error": 0.0}

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        max_err = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confs > bin_lower) & (confs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accs[in_bin])
                avg_confidence_in_bin = np.mean(confs[in_bin])
                diff = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += diff * prop_in_bin
                max_err = max(max_err, diff)

        return {
            "ece": float(round(ece, 4)),
            "max_calibration_error": float(round(max_err, 4)),
        }
