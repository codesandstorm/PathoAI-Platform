"""pathoai.scoring — Clinical sTIL Scoring Engine (Milestone 9)."""

from pathoai.core.types import ClinicalReport, STILScore
from pathoai.scoring.bootstrap import BootstrapEngine
from pathoai.scoring.categorization import STILCategorizer
from pathoai.scoring.clinical_rules import ClinicalRules
from pathoai.scoring.confidence import ConfidenceEstimator
from pathoai.scoring.explainability import STILExplainability
from pathoai.scoring.exporter import (
    export_clinical_report_to_markdown,
    export_stil_score_to_json,
)
from pathoai.scoring.factory import create_scorer
from pathoai.scoring.pipeline import ScoringPipeline
from pathoai.scoring.registry import (
    get_scorer_class,
    list_registered_scorers,
    register_scorer,
)
from pathoai.scoring.report import ReportGenerator
from pathoai.scoring.scorer import sTILScorer
from pathoai.scoring.statistics import StatisticsEngine
from pathoai.scoring.summary import generate_scoring_summary
from pathoai.scoring.validation import ScoreValidator
from pathoai.scoring.visualization import create_stil_density_heatmap

__all__ = [
    "STILScore",
    "ClinicalReport",
    "register_scorer",
    "get_scorer_class",
    "list_registered_scorers",
    "create_scorer",
    "StatisticsEngine",
    "sTILScorer",
    "BootstrapEngine",
    "ConfidenceEstimator",
    "ClinicalRules",
    "STILCategorizer",
    "STILExplainability",
    "ScoreValidator",
    "ReportGenerator",
    "ScoringPipeline",
    "export_stil_score_to_json",
    "export_clinical_report_to_markdown",
    "create_stil_density_heatmap",
    "generate_scoring_summary",
]
