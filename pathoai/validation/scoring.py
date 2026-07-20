"""
pathoai/validation/scoring.py
==============================
Clinical sTIL Scoring Evaluator.

Evaluates MAE, RMSE, Pearson r, Spearman rho, R^2, ICC, and Bland–Altman agreement metrics
between automated sTIL scores and ground-truth pathologist assessments.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.7
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from pathoai.core.types import ScoringMetrics
from pathoai.validation.agreement import AgreementEngine
from pathoai.validation.correlation import CorrelationEngine
from pathoai.validation.registry import register_evaluator


@register_evaluator("scoring")
class ClinicalScoringEvaluator:
    """Evaluates clinical scoring agreement metrics against pathologist assessments."""

    def __init__(self) -> None:
        self.corr_engine = CorrelationEngine()
        self.agreement_engine = AgreementEngine()

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> ScoringMetrics:
        """Evaluates agreement and correlation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Pathologist ground-truth sTIL scores.
        y_pred : np.ndarray
            Automated AI sTIL scores.

        Returns
        -------
        ScoringMetrics
            Calculated scoring metrics container DTO.
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        if len(y_t) == 0:
            return ScoringMetrics(
                mae=0.0, rmse=0.0, pearson_r=0.0, pearson_pvalue=1.0,
                spearman_r=0.0, spearman_pvalue=1.0, r2=0.0, icc=1.0,
                bland_altman_bias=0.0, bland_altman_lower_limit=0.0, bland_altman_upper_limit=0.0,
            )

        mae = float(np.mean(np.abs(y_p - y_t)))
        rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))

        corrs = self.corr_engine.compute_correlations(y_t, y_p)
        icc = self.agreement_engine.compute_icc(y_t, y_p)
        ba = self.agreement_engine.compute_bland_altman(y_t, y_p)

        return ScoringMetrics(
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            pearson_r=corrs["pearson_r"],
            pearson_pvalue=corrs["pearson_pvalue"],
            spearman_r=corrs["spearman_r"],
            spearman_pvalue=corrs["spearman_pvalue"],
            r2=corrs["r2"],
            icc=icc,
            bland_altman_bias=ba["bland_altman_bias"],
            bland_altman_lower_limit=ba["bland_altman_lower_limit"],
            bland_altman_upper_limit=ba["bland_altman_upper_limit"],
        )
