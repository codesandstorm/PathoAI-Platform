# REST API Specification

The REST API server ([pathoai/dashboard/api/app.py](file:///d:/Research/PathoAI-Platform/pathoai/dashboard/api/app.py)) exposes endpoints for clinical cases, slide metadata, pipeline execution, scientific validation, and experiment leaderboards.

---

## 📡 Endpoints

- `GET /api/cases`: Returns patient cases.
- `GET /api/experiments/leaderboard`: Returns experiment leaderboard.
- `POST /api/orchestrator/run`: Executes `PipelineOrchestrator.run(...)`.
