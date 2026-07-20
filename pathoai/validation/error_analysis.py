"""
pathoai/validation/error_analysis.py
====================================
Error Analysis Engine.

Identifies false positives, false negatives, failure modes, and outlier slides.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.14
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from pathoai.core.types import ErrorAnalysis


class ErrorAnalysisEngine:
    """Categorizes prediction errors and detects outlier slides."""

    def analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        slide_ids: List[str],
        error_threshold: float = 15.0,
    ) -> ErrorAnalysis:
        """Analyzes absolute errors to detect outlier slides and categorize failure modes."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        diffs = np.abs(y_p - y_t)

        outliers = []
        fp_count = 0
        fn_count = 0

        for idx, diff in enumerate(diffs):
            s_id = slide_ids[idx] if idx < len(slide_ids) else f"slide_{idx}"
            if diff > error_threshold:
                outliers.append(s_id)

            if y_p[idx] > y_t[idx]:
                fp_count += 1
            elif y_p[idx] < y_t[idx]:
                fn_count += 1

        failure_modes = {
            "over_estimation": fp_count,
            "under_estimation": fn_count,
            "extreme_outliers": len(outliers),
        }

        return ErrorAnalysis(
            false_positives_count=fp_count,
            false_negatives_count=fn_count,
            outlier_slides=outliers,
            failure_modes=failure_modes,
        )
