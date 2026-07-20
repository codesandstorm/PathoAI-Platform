"""
tests/unit/dashboard/test_api_routes.py
========================================
Unit tests for Platform REST API service and endpoints.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.dashboard.api.routes import PlatformAPIService


class TestAPIRoutes:
    """Test REST API service."""

    def test_get_cases_endpoint(self):
        """Test get_cases endpoint."""
        service = PlatformAPIService()
        cases = service.get_cases()
        assert isinstance(cases, list)
        assert len(cases) >= 1
        assert cases[0]["id"] == "CASE-2026-8891"

    def test_post_orchestrator_run_endpoint(self):
        """Test run_pipeline endpoint."""
        service = PlatformAPIService()
        resp = service.run_pipeline("test_slide_01")
        assert resp["slide_id"] == "test_slide_01"
        assert resp["score_percent"] == 28.5
