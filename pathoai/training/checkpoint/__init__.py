"""pathoai.training.checkpoint — model weights checkpointing.

Exposes:
    CheckpointManager: Class managing best/last models saving, pruning, and restore.
"""

from pathoai.training.checkpoint.manager import CheckpointManager

__all__ = [
    "CheckpointManager",
]
