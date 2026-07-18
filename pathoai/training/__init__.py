"""pathoai.training — Research-grade Training Engine package.

Exposes:
    Trainer: Model-agnostic training loop engine.
    TrainerState: Training progress container.
    Callback: Base callback class.
    EarlyStopping: Stopping criterion.
    ModelCheckpoint: Checkpoint saving.
    LRSchedulerCallback: Steps learning rate schedulers.
    ProgressLogger: tqdm console progress bar.
    SegmentationMetrics: Segmentation accuracy indices.
    ConfusionMatrixMetric: Confusion matrix and Cohen's Kappa.
    MetricCollection: Aggregates metrics update/compute/reset.
    HistoryTracker: Handles history recording and formatting.
    CheckpointManager: Manages checkpoint serialization and pruning.
    Experiment: Manages experiment directories and configuration backups.
"""

from pathoai.training.callbacks import (
    Callback,
    EarlyStopping,
    LRSchedulerCallback,
    ModelCheckpoint,
    ProgressLogger,
)
from pathoai.training.checkpoint import CheckpointManager
from pathoai.training.experiment import Experiment
from pathoai.training.history import HistoryTracker
from pathoai.training.metrics import (
    ConfusionMatrixMetric,
    MetricCollection,
    SegmentationMetrics,
)
from pathoai.training.trainer import Trainer, TrainerState

__all__ = [
    "Trainer",
    "TrainerState",
    "Callback",
    "EarlyStopping",
    "LRSchedulerCallback",
    "ModelCheckpoint",
    "ProgressLogger",
    "CheckpointManager",
    "Experiment",
    "HistoryTracker",
    "ConfusionMatrixMetric",
    "MetricCollection",
    "SegmentationMetrics",
]
