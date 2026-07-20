"""
pathoai/validation/benchmark.py
================================
Benchmarking Comparison Engine.

Compares target platform metrics against literature and baseline benchmark values.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.12
"""

from __future__ import annotations

from typing import Dict

from pathoai.core.types import BenchmarkResults


class BenchmarkEngine:
    """Evaluates percentage improvement over literature baselines."""

    def compare_to_baseline(
        self,
        target_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        baseline_name: str = "TIGER_Baseline",
    ) -> BenchmarkResults:
        """Compares target metrics against baseline metrics."""
        improvements = {}
        for key, target_val in target_metrics.items():
            base_val = baseline_metrics.get(key, 0.0)
            if base_val != 0:
                imp = ((target_val - base_val) / abs(base_val)) * 100.0
            else:
                imp = 0.0
            improvements[key] = round(imp, 2)

        return BenchmarkResults(
            baseline_name=baseline_name,
            target_metrics=target_metrics,
            baseline_metrics=baseline_metrics,
            percentage_improvements=improvements,
        )
