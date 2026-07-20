"""
pathoai/dashboard/api/schemas.py
================================
Pydantic API DTO Schemas for Clinical Digital Pathology Platform REST API.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 11.1
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClinicalCaseDTO(BaseModel):
    id: str
    patient: str
    hospital: str
    scanner: str
    diagnosis: str
    stil: float
    ci: str
    category: str
    pathologist: str
    status: str


class PipelineRunRequest(BaseModel):
    slide_id: str
    segmentation_model: str = "deeplabv3plus"
    detection_model: str = "yolo"


class ClinicalReportResponse(BaseModel):
    slide_id: str
    score_percent: float
    clinical_category: str
    confidence_interval: List[float]
    interpretation: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
