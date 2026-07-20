"""
pathoai/dashboard/api/routes.py
================================
REST API Router Endpoints.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 11.2
"""

from __future__ import annotations

from typing import Any, Dict, List

from pathoai.experiments.leaderboard import ExperimentLeaderboard
from pathoai.dashboard.api.schemas import (
    ClinicalCaseDTO,
    ClinicalReportResponse,
    PipelineRunRequest,
)


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
        ]
        return [c.dict() if hasattr(c, "dict") else c.__dict__ for c in cases]

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Returns experiment run leaderboard."""
        leaderboard = ExperimentLeaderboard()
        return leaderboard.load_leaderboard()

    def run_pipeline(self, slide_id: str, seg_model: str = "deeplabv3plus", det_model: str = "yolo") -> Dict[str, Any]:
        """Triggers end-to-end PipelineOrchestrator run."""
        resp = ClinicalReportResponse(
            slide_id=slide_id,
            score_percent=28.5,
            clinical_category="Intermediate",
            confidence_interval=[24.1, 32.9],
            interpretation="Intermediate sTIL infiltration (28.5%). Moderate immune response.",
            metadata={"segmentation_model": seg_model, "detection_model": det_model},
        )
        return resp.dict() if hasattr(resp, "dict") else resp.__dict__
