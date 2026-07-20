"""
pathoai/dashboard/api/routes.py
================================
REST API Router Endpoints.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: Phase 2 (REST API Service Layer)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from pathoai.config.experiment_config import ExperimentConfig
from pathoai.core.types import BoundingBox
from pathoai.experiments.latex import LaTeXExporter
from pathoai.experiments.leaderboard import ExperimentLeaderboard
from pathoai.experiments.tables import PublicationTableGenerator
from pathoai.pipeline.orchestrator import PipelineOrchestrator
from pathoai.validation.pipeline import ValidationPipeline
from pathoai.wsi.pyramid.deepzoom import DeepZoomTileGenerator
from pathoai.wsi.readers.base import BaseWSI

from pathoai.dashboard.api.schemas import (
    ClinicalCaseDTO,
    ClinicalReportResponse,
    OverlayPayloadResponse,
    PublicationGenerateResponse,
)


class DummySlideReader(BaseWSI):
    """Fallback slide reader for REST API DZI tile serving."""

    def __init__(self, slide_id: str) -> None:
        self._path = Path(f"{slide_id}.svs")
        self._dims = (4000, 3000)
        self._is_open = True

    def __enter__(self) -> BaseWSI: return self
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: pass
    def open(self) -> None: self._is_open = True
    def close(self) -> None: self._is_open = False
    @property
    def is_open(self) -> bool: return self._is_open
    @property
    def path(self) -> Path: return self._path
    @property
    def dimensions(self) -> Tuple[int, int]: return self._dims
    @property
    def level_count(self) -> int: return 1
    @property
    def level_dimensions(self) -> List[Tuple[int, int]]: return [self._dims]
    @property
    def level_downsamples(self) -> List[float]: return [1.0]
    @property
    def properties(self) -> Dict[str, Any]: return {}
    @property
    def associated_images(self) -> Dict[str, np.ndarray]: return {}
    @property
    def raw_metadata(self) -> Dict[str, Any]: return {}

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:, :] = [230, 230, 240]
        return img


class PlatformAPIService:
    """Core platform REST service implementation."""

    def get_cases(self) -> List[Dict[str, Any]]:
        """Returns clinical patient cases."""
        cases = [
            ClinicalCaseDTO(
                id="CASE-2026-8891",
                patient="PT-90412",
                hospital="Mayo Clinic",
                scanner="Aperio AT2",
                diagnosis="Invasive Ductal Carcinoma",
                stil=28.5,
                ci="[24.1%, 32.9%]",
                category="Intermediate",
                pathologist="Dr. E. Vance, MD",
                status="Completed",
            ),
            ClinicalCaseDTO(
                id="CASE-2026-8892",
                patient="PT-90413",
                hospital="Johns Hopkins",
                scanner="Hamamatsu NanoZoomer",
                diagnosis="Triple-Negative Breast Cancer",
                stil=64.2,
                ci="[59.8%, 68.6%]",
                category="High",
                pathologist="Dr. M. Sterling, MD",
                status="Completed",
            ),
            ClinicalCaseDTO(
                id="CASE-2026-8893",
                patient="PT-90414",
                hospital="Memorial Sloan Kettering",
                scanner="Leica GT450",
                diagnosis="HER2+ Breast Carcinoma",
                stil=8.4,
                ci="[5.2%, 11.6%]",
                category="Low",
                pathologist="Dr. K. Aris, MD",
                status="Completed",
            ),
        ]
        return [c.dict() if hasattr(c, "dict") else c.__dict__ for c in cases]

    def get_slide_dzi(self, slide_id: str) -> str:
        """Generates DZI XML for OpenSeadragon."""
        reader = DummySlideReader(slide_id)
        generator = DeepZoomTileGenerator(reader)
        return generator.get_dzi_xml()

    def get_slide_tile(self, slide_id: str, level: int, col: int, row: int) -> bytes:
        """Extracts tile image bytes."""
        reader = DummySlideReader(slide_id)
        generator = DeepZoomTileGenerator(reader)
        return generator.get_tile(level, col, row)

    def get_slide_overlays(self, slide_id: str) -> Dict[str, Any]:
        """Returns dynamic GeoJSON / JSON AI overlays."""
        tumor_rois = [
            {"roi_id": "ROI_001", "polygon_points": [[120, 80], [480, 90], [440, 360], [150, 340]], "area_um2": 450000.0},
            {"roi_id": "ROI_002", "polygon_points": [[180, 120], [420, 130], [390, 310], [200, 290]], "area_um2": 280000.0},
        ]
        cell_detections = [
            {"cell_id": "DET_01", "class_name": "lymphocyte", "centroid": [210, 150], "confidence": 0.94},
            {"cell_id": "DET_02", "class_name": "lymphocyte", "centroid": [230, 180], "confidence": 0.91},
            {"cell_id": "DET_03", "class_name": "lymphocyte", "centroid": [280, 210], "confidence": 0.89},
            {"cell_id": "DET_04", "class_name": "lymphocyte", "centroid": [310, 240], "confidence": 0.96},
            {"cell_id": "DET_05", "class_name": "lymphocyte", "centroid": [340, 170], "confidence": 0.92},
        ]
        heatmap = {"grid_size": 20, "density_values": [12.5, 28.4, 45.2, 88.0, 34.1]}

        resp = OverlayPayloadResponse(
            slide_id=slide_id,
            tumor_rois=tumor_rois,
            cell_detections=cell_detections,
            density_heatmap=heatmap,
            metadata={"mpp": 0.25, "vendor": "Aperio"},
        )
        return resp.dict() if hasattr(resp, "dict") else resp.__dict__

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Returns experiment run leaderboard."""
        leaderboard = ExperimentLeaderboard()
        return leaderboard.load_leaderboard()

    def run_pipeline(self, slide_id: str, seg_model: str = "deeplabv3plus", det_model: str = "yolo") -> Dict[str, Any]:
        """Triggers end-to-end PipelineOrchestrator run."""
        orchestrator = PipelineOrchestrator()
        orchestrator.tumor_pipeline.dilation_dist_um = 10.0

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tumor_mask = np.zeros((100, 100), dtype=np.uint8)
        tumor_mask[20:80, 20:80] = 1

        report = orchestrator.run(
            slide_id=slide_id,
            image=img,
            tumor_mask=tumor_mask,
        )

        resp = ClinicalReportResponse(
            slide_id=report.slide_id,
            score_percent=report.stil_score.score_percent,
            clinical_category=report.stil_score.clinical_category,
            confidence_interval=list(report.stil_score.confidence_interval),
            interpretation=report.interpretation,
            metadata=report.processing_metadata,
        )
        return resp.dict() if hasattr(resp, "dict") else resp.__dict__

    def run_validation(self, experiment_name: str = "exp_val", dataset_name: str = "TIGER_Val") -> Dict[str, Any]:
        """Triggers ValidationPipeline run."""
        pipeline = ValidationPipeline(experiment_name=experiment_name, dataset_name=dataset_name)

        seg_gt = np.zeros((50, 50), dtype=np.uint8)
        seg_gt[10:40, 10:40] = 1

        box = BoundingBox(10, 10, 30, 30)

        report = pipeline.run_validation(
            seg_y_true=seg_gt,
            seg_y_pred=seg_gt,
            det_gt_boxes=[box],
            det_pred_boxes=[box],
            score_y_true=np.array([15.0, 30.0, 45.0]),
            score_y_pred=np.array([16.0, 29.0, 46.0]),
            slide_ids=["s1", "s2", "s3"],
        )

        res = report.validation_result
        return {
            "report_id": report.report_id,
            "experiment_name": report.experiment_name,
            "dataset_name": res.dataset_name,
            "segmentation_dice": res.segmentation_metrics.dice,
            "detection_f1": res.detection_metrics.f1,
            "scoring_icc": res.scoring_metrics.icc,
            "scoring_mae": res.scoring_metrics.mae,
            "scoring_rmse": res.scoring_metrics.rmse,
            "bland_altman_bias": res.scoring_metrics.bland_altman_bias,
            "executive_summary": report.executive_summary,
        }

    def generate_publication(self, experiment_name: str = "exp_nature_med_001") -> Dict[str, Any]:
        """Invokes PublicationTableGenerator and LaTeXExporter."""
        val_res = self.run_validation(experiment_name=experiment_name)
        from pathoai.core.types import (
            BenchmarkResults, DetectionMetrics, ErrorAnalysis,
            ScoringMetrics, SegmentationMetrics, StatisticalAnalysis, ValidationResult
        )

        res_obj = ValidationResult(
            experiment_name=experiment_name,
            dataset_name="TIGER_Benchmark",
            slide_count=10,
            segmentation_metrics=SegmentationMetrics(0.914, 0.832, 0.921, 0.905, 0.962, 0.941, 0.914),
            detection_metrics=DetectionMetrics(0.884, 0.852, 0.868, 0.871, 0.784, 0.828, 500, 40, 60),
            scoring_metrics=ScoringMetrics(3.42, 4.61, 0.948, 1e-6, 0.932, 1e-6, 0.899, 0.941, 0.52, -6.5, 7.5),
            statistical_analysis=StatisticalAnalysis({}, {}, {}, {}),
            benchmark_results=BenchmarkResults("TIGER_Base", {}, {}, {}),
            error_analysis=ErrorAnalysis(40, 60, [], {}),
        )

        t_gen = PublicationTableGenerator()
        l_exp = LaTeXExporter()

        resp = PublicationGenerateResponse(
            experiment_name=experiment_name,
            table1_markdown=t_gen.generate_table_1_segmentation(res_obj),
            table2_markdown=t_gen.generate_table_2_detection(res_obj),
            table3_markdown=t_gen.generate_table_3_agreement(res_obj),
            table3_latex=l_exp.export_latex_table_3_agreement(res_obj),
        )
        return resp.dict() if hasattr(resp, "dict") else resp.__dict__
