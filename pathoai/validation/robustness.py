"""
pathoai/validation/robustness.py
================================
Pipeline Robustness Evaluator.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.11
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class RobustnessEngine:
    """Evaluates stability across operational variations."""

    def evaluate_magnification_robustness(
        self, scores_20x: List[float], scores_40x: List[float]
    ) -> Dict[str, float]:
        """Evaluates agreement between 20x and 40x sTIL scores."""
        s20 = np.asarray(scores_20x)
        s40 = np.asarray(scores_40x)
        if len(s20) == 0:
            return {"mean_difference": 0.0, "mae_magnification": 0.0}

        diff = float(np.mean(np.abs(s40 - s20)))
        return {
            "mean_difference": float(round(np.mean(s40 - s20), 4)),
            "mae_magnification": float(round(diff, 4)),
        }
