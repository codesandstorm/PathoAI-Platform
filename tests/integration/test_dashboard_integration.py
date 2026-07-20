"""
tests/integration/test_dashboard_integration.py
================================================
Integration tests for Clinical Digital Pathology Platform REST API.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.dashboard.api.routes import PlatformAPIService


def test_end_to_end_dashboard_api_integration():
    """Verifies complete REST API server routing and service methods."""
    service = PlatformAPIService()

    cases = service.get_cases()
    assert len(cases) >= 2

    leaderboard = service.get_leaderboard()
    assert isinstance(leaderboard, list)

    run_res = service.run_pipeline("slide_integration_01")
    assert run_res["clinical_category"] == "Intermediate"
    assert run_res["slide_id"] == "slide_integration_01"
