"""
pathoai/validation/pipeline.py
==============================
Validation Pipeline Coordinator.

Orchestrates multi-stage evaluation across segmentation, cell detection, spatial fusion,
and clinical scoring, producing typed ValidationResult and ValidationReport DTOs.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.20
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from pathoai.core.types import BoundingBox, ValidationReport, ValidationResult
from pathoai.validation.benchmark import BenchmarkEngine
from pathoai.validation.detection import DetectionEvaluator
from pathoai.validation.error_analysis import ErrorAnalysisEngine
from pathoai.validation.report import ValidationReportGenerator
from pathoai.validation.scoring import ClinicalScoringEvaluator
from pathoai.validation.segmentation import SegmentationEvaluator
from pathoai.validation.statistics import ValidationStatistics


class ValidationPipeline:
    """Coordinating master pipeline for platform scientific validation."""

    def __init__(self, experiment_name: str = "exp_validation", dataset_name: str = "TIGER_Val") -> None:
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.seg_evaluator = SegmentationEvaluator()
        self.det_evaluator = DetectionEvaluator()
        self.scoring_evaluator = ClinicalScoringEvaluator()
        self.stats_engine = ValidationStatistics()
        self.benchmark_engine = BenchmarkEngine()
        self.error_engine = ErrorAnalysisEngine()
        self.report_generator = ValidationReportGenerator()

    def run_validation(
        self,
        seg_y_true: np.ndarray,
        seg_y_pred: np.ndarray,
        det_gt_boxes: List[BoundingBox],
        det_pred_boxes: List[BoundingBox],
        score_y_true: np.ndarray,
        score_y_pred: np.ndarray,
        slide_ids: List[str],
        baseline_scores: Optional[Dict[str, float]] = None,
    ) -> ValidationReport:
        """Executes complete multi-stage validation evaluation.

        Parameters
        ----------
        seg_y_true : np.ndarray
            Ground-truth segmentation mask.
        seg_y_pred : np.ndarray
            Predicted segmentation mask.
        det_gt_boxes : List[BoundingBox]
            Ground-truth detection boxes.
        det_pred_boxes : List[BoundingBox]
            Predicted detection boxes.
        score_y_true : np.ndarray
            Pathologist ground-truth sTIL scores.
        score_y_pred : np.ndarray
            AI model sTIL scores.
        slide_ids : List[str]
            List of slide IDs.
        baseline_scores : Optional[Dict[str, float]]
            Literature baseline benchmark metrics.

        Returns
        -------
        ValidationReport
            Executive validation report DTO.
        """
        # 1. Stage evaluations
        seg_metrics = self.seg_evaluator.evaluate(seg_y_true, seg_y_pred)
        det_metrics = self.det_evaluator.evaluate(det_gt_boxes, det_pred_boxes)
        scoring_metrics = self.scoring_evaluator.evaluate(score_y_true, score_y_pred)

        # 2. Statistics & Bootstrap
        stats_analysis = self.stats_engine.compute_statistical_analysis(score_y_true, score_y_pred)

        # 3. Benchmarking
        target_dict = {
            "dice": seg_metrics.dice,
            "f1_detection": det_metrics.f1,
            "icc": scoring_metrics.icc,
            "mae": scoring_metrics.mae,
        }
        base_dict = baseline_scores or {"dice": 0.75, "f1_detection": 0.70, "icc": 0.80, "mae": 12.0}
        benchmark_results = self.benchmark_engine.compare_to_baseline(target_dict, base_dict)

        # 4. Error Analysis
        error_analysis = self.error_engine.analyze_errors(score_y_true, score_y_pred, slide_ids)

        # 5. Master ValidationResult DTO
        val_result = ValidationResult(
            experiment_name=self.experiment_name,
            dataset_name=self.dataset_name,
            slide_count=len(slide_ids) if slide_ids else 1,
            segmentation_metrics=seg_metrics,
            detection_metrics=det_metrics,
            scoring_metrics=scoring_metrics,
            statistical_analysis=stats_analysis,
            benchmark_results=benchmark_results,
            error_analysis=error_analysis,
        )

        # 6. ValidationReport DTO
        return self.report_generator.generate_report(val_result)
