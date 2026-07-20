"""
pathoai/experiments/leaderboard.py
===================================
Experiment Leaderboard Engine.

Ranks and compares historical experiment runs logged under experiments/.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5.8
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


class ExperimentLeaderboard:
    """Aggregates and ranks historical experiment runs."""

    def __init__(self, base_dir: Union[str, Path] = "experiments") -> None:
        self.base_dir = Path(base_dir)

    def load_leaderboard(self) -> List[Dict[str, Any]]:
        """Loads all metrics.json files across experiment directories and ranks by ICC."""
        runs = []
        if not self.base_dir.exists():
            return runs

        for exp_path in self.base_dir.iterdir():
            if exp_path.is_dir():
                metrics_file = exp_path / "metrics.json"
                if metrics_file.is_file():
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        runs.append(data)

        # Sort by ICC descending
        runs.sort(key=lambda x: x.get("icc", 0.0), reverse=True)
        return runs
