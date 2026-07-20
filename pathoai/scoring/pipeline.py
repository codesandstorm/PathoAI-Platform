"""
pathoai/scoring/pipeline.py
===========================
Clinical Scoring Pipeline Coordinator.

Orchestrates execution workflow from FusionResult inputs to STILScore and
ClinicalReport domain outputs.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.16
"""

from __future__ import annotations

from typing import Any, Optional

from pathoai.core.types import ClinicalReport, FusionResult, STILScore
from pathoai.scoring.bootstrap import BootstrapEngine
from pathoai.scoring.categorization import STILCategorizer
from pathoai.scoring.clinical_rules import ClinicalRules
from pathoai.scoring.explainability import STILExplainability
from pathoai.scoring.factory import create_scorer
from pathoai.scoring.report import ReportGenerator
from pathoai.scoring.scorer import sTILScorer
from pathoai.scoring.statistics import StatisticsEngine
from pathoai.scoring.validation import ScoreValidator


class ScoringPipeline:
    """Coordinating pipeline for clinical sTIL scoring."""

    def __init__(
        self,
        config: Optional[Any] = None,
        lymphocyte_diameter_um: float = 10.0,
        n_bootstrap_iterations: int = 500,
        low_threshold: float = 10.0,
        high_threshold: float = 50.0,
    ) -> None:
        """
        Parameters
        ----------
        config : Optional[Any]
            Config object.
        lymphocyte_diameter_um : float
            Lymphocyte diameter in microns.
        n_bootstrap_iterations : int
            Number of bootstrap iterations for 95% CI calculation.
        low_threshold : float
            Threshold for Low sTIL category.
        high_threshold : float
            Threshold for High sTIL category.
        """
        if config is not None:
            scorer = create_scorer(config)
            lymphocyte_diameter_um = scorer.lymphocyte_diameter_um

        self.scorer = sTILScorer(lymphocyte_diameter_um=lymphocyte_diameter_um)
        self.stats_engine = StatisticsEngine()
        self.bootstrap_engine = BootstrapEngine(n_iterations=n_bootstrap_iterations)
        self.rules = ClinicalRules(low_threshold=low_threshold, high_threshold=high_threshold)
        self.categorizer = STILCategorizer(rules=self.rules)
        self.explainability = STILExplainability()
        self.validator = ScoreValidator()
        self.report_generator = ReportGenerator(rules=self.rules)

    def process(self, fusion_result: FusionResult) -> ClinicalReport:
        """Executes clinical sTIL scoring workflow on a FusionResult.

        Parameters
        ----------
        fusion_result : FusionResult
            Spatial fusion result container.

        Returns
        -------
        ClinicalReport
            Comprehensive clinical report containing STILScore DTO.
        """
        # 1. Statistics
        stats = self.stats_engine.compute_statistics(fusion_result)

        # 2. Score calculation
        score_percent = self.scorer.compute_stil_score_percent(fusion_result)

        # 3. Bootstrap CI
        ci = self.bootstrap_engine.compute_confidence_interval(fusion_result, score_percent)

        # 4. Categorization & Explainability
        category = self.categorizer.categorize(score_percent)
        explanation = self.explainability.generate_explanation(score_percent, stats, ci, category)

        # 5. Assemble STILScore DTO
        stil_score = STILScore(
            slide_id=fusion_result.slide_id,
            score_percent=score_percent,
            stromal_area_mm2=stats["stromal_area_mm2"],
            stromal_lymphocytes=stats["stromal_lymphocytes"],
            lymphocyte_density=stats["lymphocyte_density_per_mm2"],
            confidence_interval=ci,
            confidence_level=0.95,
            clinical_category=category,
            explanation=explanation,
            metadata={"n_bootstrap_iterations": self.bootstrap_engine.n_iterations},
        )

        # 6. Validate STILScore
        val = self.validator.validate_score(stil_score)
        if not val["passed"]:
            raise ValueError(f"Score validation failed: {val['issues']}")

        # 7. Generate ClinicalReport
        return self.report_generator.generate_report(stil_score)
