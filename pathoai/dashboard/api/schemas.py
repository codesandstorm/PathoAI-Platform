"""
pathoai/dashboard/api/schemas.py
================================
Pydantic API DTO Schemas for Clinical Digital Pathology Platform REST API.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: Phase 2 (REST API Integration)
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


class ValidationRunRequest(BaseModel):
    experiment_name: str = "exp_clinical_val"
    dataset_name: str = "TIGER_Val"


class ClinicalReportResponse(BaseModel):
    slide_id: str
    score_percent: float
    clinical_category: str
    confidence_interval: List[float]
    interpretation: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OverlayPayloadResponse(BaseModel):
    slide_id: str
    tumor_rois: List[Dict[str, Any]]
    cell_detections: List[Dict[str, Any]]
    density_heatmap: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PublicationGenerateResponse(BaseModel):
    experiment_name: str
    table1_markdown: str
    table2_markdown: str
    table3_markdown: str
    table3_latex: str
