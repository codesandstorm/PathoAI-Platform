"""pathoai.training.trainer — core model-agnostic training loop engine.

Exposes:
    Trainer: Class orchestrating PyTorch forward/backward passes and evaluation loops.
    TrainerState: Dataclass tracking training progression metrics.
"""

from pathoai.training.trainer.state import TrainerState
from pathoai.training.trainer.trainer import Trainer

__all__ = [
    "Trainer",
    "TrainerState",
]
