"""
tests/integration/test_dashboard_integration.py
================================================
Integration tests for Clinical Digital Pathology Platform REST API.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.dashboard.api.routes import PlatformAPIService


def test_end_to_end_dashboard_api_integration():
    """Verifies complete REST API server routing, DeepZoom tiles, AI overlays, and validation integration."""
    service = PlatformAPIService()

    cases = service.get_cases()
    assert len(cases) >= 2

    dzi_xml = service.get_slide_dzi("CASE-2026-8891")
    assert 'xmlns="http://schemas.microsoft.com/deepzoom/2008"' in dzi_xml

    tile_bytes = service.get_slide_tile("CASE-2026-8891", 10, 0, 0)
    assert tile_bytes[:4] == b"\x89PNG"

    overlays = service.get_slide_overlays("CASE-2026-8891")
    assert len(overlays["tumor_rois"]) >= 1
    assert "checkpoint" in overlays["metadata"]
    assert "run_id" in overlays["metadata"]

    val_res = service.run_validation("exp_nature_med_001", "TIGER_Grand_Challenge")
    assert val_res["scoring_icc"] > 0.9

    pub_res = service.generate_publication("exp_nature_med_001")
    assert "Table 3:" in pub_res["table3_markdown"]
