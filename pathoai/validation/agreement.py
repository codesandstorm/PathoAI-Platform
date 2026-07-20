"""
pathoai/validation/agreement.py
===============================
Clinical Inter-Rater Agreement Engine.

Computes Intraclass Correlation Coefficient (ICC) and Bland–Altman limits of agreement
(mean bias, lower limit, upper limit).

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.6
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class AgreementEngine:
    """Computes ICC agreement and Bland–Altman statistics."""

    def compute_icc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Intraclass Correlation Coefficient (ICC).

        Parameters
        ----------
        y_true : np.ndarray
            Rater 1 scores (Pathologist).
        y_pred : np.ndarray
            Rater 2 scores (AI model).

        Returns
        -------
        float
            ICC score in range [0, 1].
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        n = len(y_t)
        if n < 2:
            return 1.0

        mean_t = np.mean(y_t)
        mean_p = np.mean(y_p)
        grand_mean = (mean_t + mean_p) / 2.0

        # Between-subject variance
        row_means = (y_t + y_p) / 2.0
        ss_between = 2.0 * np.sum((row_means - grand_mean) ** 2)

        # Within-subject variance
        ss_within = np.sum((y_t - row_means) ** 2 + (y_p - row_means) ** 2)

        ms_between = ss_between / (n - 1) if (n - 1) > 0 else 0.0
        ms_within = ss_within / n if n > 0 else 0.0

        if (ms_between + ms_within) <= 0:
            return 1.0

        icc = (ms_between - ms_within) / (ms_between + ms_within)
        return float(round(max(0.0, min(1.0, icc)), 4))

    def compute_bland_altman(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Computes Bland–Altman mean bias and 95% limits of agreement.

        Parameters
        ----------
        y_true : np.ndarray
            Rater 1 scores.
        y_pred : np.ndarray
            Rater 2 scores.

        Returns
        -------
        Dict[str, float]
            Bland–Altman statistics dictionary.
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        diffs = y_p - y_t

        mean_bias = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

        lower_limit = mean_bias - 1.96 * std_diff
        upper_limit = mean_bias + 1.96 * std_diff

        return {
            "bland_altman_bias": float(round(mean_bias, 4)),
            "bland_altman_lower_limit": float(round(lower_limit, 4)),
            "bland_altman_upper_limit": float(round(upper_limit, 4)),
            "sd_diff": float(round(std_diff, 4)),
        }
