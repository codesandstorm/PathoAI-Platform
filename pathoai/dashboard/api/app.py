"""
pathoai/dashboard/api/app.py
=============================
Application Entrypoint for Clinical Digital Pathology Platform.

Supports FastAPI when installed, and provides a lightweight WSGI/Python app fallback.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 11.3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pathoai.dashboard.api.routes import PlatformAPIService

service = PlatformAPIService()

try:
    from fastapi import FastAPI
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

    @app.get("/api/experiments/leaderboard")
    def api_get_leaderboard():
        return service.get_leaderboard()

    @app.post("/api/orchestrator/run")
    def api_run_pipeline(data: Dict[str, Any]):
        slide_id = data.get("slide_id", "slide_01")
        return service.run_pipeline(slide_id)

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

        def get_leaderboard(self) -> List[Dict[str, Any]]:
            return self.service.get_leaderboard()

        def run_pipeline(self, slide_id: str) -> Dict[str, Any]:
            return self.service.run_pipeline(slide_id)

    app = FallbackApp()
