"""
pathoai/training/trainer/state.py
================================
State container for the model training lifecycle.

Tracks the progress metrics (epochs, step counts, losses, elapsed times) during
training runs and supports checkpoint serialization.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.1
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class TrainerState:
    """Dataclass tracking the current state of a Trainer run.

    Provides a clean object to save/restore during checkpointing.
    """
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -float("inf")
    learning_rate: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert state properties to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainerState:
        """Create a TrainerState instance from a dictionary."""
        return cls(
            epoch=data.get("epoch", 0),
            global_step=data.get("global_step", 0),
            best_metric=data.get("best_metric", -float("inf")),
            learning_rate=data.get("learning_rate", 0.0),
            train_loss=data.get("train_loss", 0.0),
            val_loss=data.get("val_loss", 0.0),
            elapsed_time=data.get("elapsed_time", 0.0),
        )
