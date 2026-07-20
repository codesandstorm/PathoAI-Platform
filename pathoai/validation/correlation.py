"""
pathoai/validation/correlation.py
=================================
Statistical Correlation Engine.

Computes Pearson r, Spearman rank rho, p-values, and R^2 goodness of fit.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.stats as stats


class CorrelationEngine:
    """Computes statistical correlation and regression metrics."""

    def compute_correlations(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculates Pearson, Spearman, and R^2 metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth numerical values (e.g. Pathologist sTIL scores).
        y_pred : np.ndarray
            Predicted numerical values (e.g. AI sTIL scores).

        Returns
        -------
        Dict[str, float]
            Correlation metrics dictionary.
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        if len(y_t) < 2 or np.all(y_t == y_t[0]) or np.all(y_p == y_p[0]):
            return {
                "pearson_r": 0.0,
                "pearson_pvalue": 1.0,
                "spearman_r": 0.0,
                "spearman_pvalue": 1.0,
                "r2": 0.0,
            }

        pearson_res = stats.pearsonr(y_t, y_p)
        spearman_res = stats.spearmanr(y_t, y_p)

        # R^2 calculation
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "pearson_r": float(round(pearson_res.statistic, 4)),
            "pearson_pvalue": float(round(pearson_res.pvalue, 6)),
            "spearman_r": float(round(spearman_res.statistic, 4)),
            "spearman_pvalue": float(round(spearman_res.pvalue, 6)),
            "r2": float(round(r2, 4)),
        }
