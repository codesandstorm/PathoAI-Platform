"""
pathoai/training/run.py
=======================
PathoAI Training Pipeline Runner CLI.

Lightweight entry point for the training execution. Loads configurations,
resolves the TrainingOrchestrator pipeline orchestrator, and triggers runs.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.5
"""

from __future__ import annotations

import argparse
import sys
from pathoai.core.config import ConfigManager
from pathoai.core.logger import get_logger
from pathoai.training.orchestrator import TrainingOrchestrator

logger = get_logger("pathoai.training.run")


def run_experiment(config_path: str) -> None:
    """Wrapper that resolves config and delegates run to TrainingOrchestrator."""
    ConfigManager._config_node = None
    ConfigManager._instance = None
    ConfigManager.initialize(base_config=config_path)
    config = ConfigManager.get_instance()

    orchestrator = TrainingOrchestrator(config, config_path=config_path)
    orchestrator.run()


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(description="PathoAI Segmentation Training Pipeline Runner.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Path to YAML configuration settings file.",
    )
    args = parser.parse_args()

    try:
        run_experiment(args.config)
        sys.exit(0)
    except Exception as exc:
        logger.critical("Fatal error in training runner: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
