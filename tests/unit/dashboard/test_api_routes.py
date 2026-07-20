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
        service = PlatformAPIService()
        cases = service.get_cases()
        assert isinstance(cases, list)
        assert len(cases) >= 1
        assert cases[0]["id"] == "CASE-2026-8891"

    def test_get_slide_dzi(self):
        service = PlatformAPIService()
        xml = service.get_slide_dzi("slide_01")
        assert "<Image" in xml
        assert 'Format="png"' in xml

    def test_get_slide_tile(self):
        service = PlatformAPIService()
        tile_bytes = service.get_slide_tile("slide_01", 10, 0, 0)
        assert isinstance(tile_bytes, bytes)
        assert len(tile_bytes) > 0

    def test_get_slide_overlays(self):
        service = PlatformAPIService()
        overlays = service.get_slide_overlays("slide_01")
        assert overlays["slide_id"] == "slide_01"
        assert "tumor_rois" in overlays
        assert "cell_detections" in overlays
        assert "model_version" in overlays["metadata"]
        assert "checkpoint" in overlays["metadata"]

    def test_run_validation(self):
        service = PlatformAPIService()
        val_res = service.run_validation("exp_val", "TIGER_Val")
        assert val_res["experiment_name"] == "exp_val"
        assert val_res["scoring_icc"] > 0.9

    def test_generate_publication(self):
        service = PlatformAPIService()
        pub_res = service.generate_publication("exp_nature_med_001")
        assert pub_res["experiment_name"] == "exp_nature_med_001"
        assert "\\begin{table}" in pub_res["table3_latex"]
