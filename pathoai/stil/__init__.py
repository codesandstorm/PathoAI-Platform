"""pathoai.stil — sTIL scoring, bootstrap, confidence, and slide scoring engine."""

from pathoai.stil.aggregator import PatchAggregator
from pathoai.stil.bootstrap import calculate_bootstrap_ci
from pathoai.stil.confidence import assign_quality_flags
from pathoai.stil.engine import FusionEngine
from pathoai.stil.scorer import compute_stil_score

__all__ = [
    "compute_stil_score",
    "PatchAggregator",
    "calculate_bootstrap_ci",
    "assign_quality_flags",
    "FusionEngine",
]
