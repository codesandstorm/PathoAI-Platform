"""
tests/unit/dashboard/test_api_schemas.py
=========================================
Unit tests for API Pydantic schemas.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.dashboard.api.schemas import ClinicalCaseDTO, PipelineRunRequest


class TestAPISchemas:
    """Test API request/response DTO schemas."""

    def test_clinical_case_dto(self):
        """Test ClinicalCaseDTO serialization."""
        case = ClinicalCaseDTO(
            id="CASE-001",
            patient="PT-100",
            hospital="Mayo Clinic",
            scanner="Aperio",
            diagnosis="Carcinoma",
            stil=25.0,
            ci="[20%, 30%]",
            category="Intermediate",
            pathologist="Dr. Smith",
            status="Completed",
        )
        assert case.id == "CASE-001"
        assert case.stil == 25.0
