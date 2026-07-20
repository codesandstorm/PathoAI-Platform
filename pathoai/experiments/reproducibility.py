"""
pathoai/experiments/reproducibility.py
======================================
Reproducibility Manager Engine.

Sets master PRNG seeds across Python, NumPy, and PyTorch for deterministic experiments.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5.3
"""

from __future__ import annotations

import random

import numpy as np
import torch


class ReproducibilityManager:
    """Enforces random seed determinism across frameworks."""

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Sets random seed for random, numpy, and torch.

        Parameters
        ----------
        seed : int
            Master integer seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
