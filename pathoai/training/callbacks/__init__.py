"""pathoai.training.callbacks — Trainer lifecycle observer hooks.

Exposes:
    Callback: Base class for implementing custom training listeners.
    EarlyStopping: Stops training when monitored metrics plateau.
    ModelCheckpoint: Automatically saves weights during training.
    LRSchedulerCallback: Steps learning rate schedulers.
    ProgressLogger: Renders console progress bars.
    MetricsCallback: Aggregates validation metrics.
"""

from pathoai.training.callbacks.base import Callback
from pathoai.training.callbacks.early_stopping import EarlyStopping
from pathoai.training.callbacks.lr_scheduler import LRSchedulerCallback
from pathoai.training.callbacks.model_checkpoint import ModelCheckpoint
from pathoai.training.callbacks.progress import ProgressLogger
from pathoai.training.callbacks.metrics import MetricsCallback

__all__ = [
    "Callback",
    "EarlyStopping",
    "LRSchedulerCallback",
    "ModelCheckpoint",
    "ProgressLogger",
    "MetricsCallback",
]
