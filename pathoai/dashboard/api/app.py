"""
pathoai/dashboard/api/app.py
=============================
Application Entrypoint for Clinical Digital Pathology Platform.

Supports FastAPI when installed, and provides a lightweight WSGI/Python app fallback.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: Phase 2 (REST API Application Entrypoint)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pathoai.dashboard.api.routes import PlatformAPIService

service = PlatformAPIService()

try:
    from fastapi import FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(
        title="PathoAI Clinical Digital Pathology Platform API",
        description="Enterprise REST API backend for Whole Slide Image AI analysis, sTIL scoring, validation, and experiment tracking.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/cases")
    def api_get_cases():
        return service.get_cases()

    @app.get("/api/slides/{slide_id}/dzi")
    def api_get_slide_dzi(slide_id: str):
        xml_content = service.get_slide_dzi(slide_id)
        return Response(content=xml_content, media_type="application/xml")

    @app.get("/api/slides/{slide_id}/tiles/{level}/{col}_{row}.png")
    def api_get_slide_tile(slide_id: str, level: int, col: int, row: int):
        tile_bytes = service.get_slide_tile(slide_id, level, col, row)
        return Response(content=tile_bytes, media_type="image/png")

    @app.get("/api/slides/{slide_id}/overlays")
    def api_get_slide_overlays(slide_id: str):
        return service.get_slide_overlays(slide_id)

    @app.get("/api/experiments/leaderboard")
    def api_get_leaderboard():
        return service.get_leaderboard()

    @app.post("/api/orchestrator/run")
    def api_run_pipeline(data: Dict[str, Any]):
        slide_id = data.get("slide_id", "slide_01")
        return service.run_pipeline(slide_id)

    @app.post("/api/validation/run")
    def api_run_validation(data: Dict[str, Any]):
        exp_name = data.get("experiment_name", "exp_val")
        ds_name = data.get("dataset_name", "TIGER_Val")
        return service.run_validation(exp_name, ds_name)

    @app.post("/api/publication/generate")
    def api_generate_publication(data: Dict[str, Any]):
        exp_name = data.get("experiment_name", "exp_nature_med_001")
        return service.generate_publication(exp_name)

    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    if ui_dir.exists():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

except ImportError:
    # Fallback Application Interface for environments without FastAPI
    class FallbackApp:
        def __init__(self) -> None:
            self.service = service

        def get_cases(self) -> List[Dict[str, Any]]:
            return self.service.get_cases()

        def get_slide_dzi(self, slide_id: str) -> str:
            return self.service.get_slide_dzi(slide_id)

        def get_slide_tile(self, slide_id: str, level: int, col: int, row: int) -> bytes:
            return self.service.get_slide_tile(slide_id, level, col, row)

        def get_slide_overlays(self, slide_id: str) -> Dict[str, Any]:
            return self.service.get_slide_overlays(slide_id)

        def get_leaderboard(self) -> List[Dict[str, Any]]:
            return self.service.get_leaderboard()

        def run_pipeline(self, slide_id: str) -> Dict[str, Any]:
            return self.service.run_pipeline(slide_id)

        def run_validation(self, experiment_name: str = "exp_val", dataset_name: str = "TIGER_Val") -> Dict[str, Any]:
            return self.service.run_validation(experiment_name, dataset_name)

        def generate_publication(self, experiment_name: str = "exp_nature_med_001") -> Dict[str, Any]:
            return self.service.generate_publication(experiment_name)

    app = FallbackApp()
