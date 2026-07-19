"""
pathoai/training/checkpoint/manager.py
======================================
Checkpoint Manager for PathoAI models.

Manages saving and loading model weights, optimizer states, and training progress.
Tracks best checkpoints, saves last checkpoints, manages top-K epoch weights,
and supports resuming training runs.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.6
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pathoai.core.logger import get_logger
from pathoai.training.trainer.state import TrainerState

logger = get_logger(__name__)


class CheckpointManager:
    """Manages model checkpoint saving, loading, top-K selection, and pruning."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_dice",
        mode: str = "max",
        save_top_k: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        checkpoint_dir : str | Path
            Directory where checkpoints will be saved.
        monitor : str
            Metric key monitored to determine the best models.
        mode : str
            Optimization mode: 'min' or 'max'.
        save_top_k : int
            Number of best checkpoints to preserve. Older files will be pruned.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max'. Got: {mode}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k

        self.best_path = self.checkpoint_dir / "best.pt"
        self.last_path = self.checkpoint_dir / "last.pt"
        self.meta_path = self.checkpoint_dir / "checkpoints_metadata.json"

        # List of dict: [{"path": str, "value": float, "epoch": int}]
        self.top_k_checkpoints: List[Dict[str, Any]] = []

        # Load metadata if it exists
        self._load_metadata()

    @property
    def best_epoch(self) -> Optional[int]:
        """Return the epoch number of the best checkpoint."""
        if not self.top_k_checkpoints:
            return None
        return self.top_k_checkpoints[0]["epoch"]

    def _load_metadata(self) -> None:
        """Load checkpoint tracking metadata from disk."""
        if self.meta_path.is_file():
            try:
                with open(self.meta_path, encoding="utf-8") as f:
                    self.top_k_checkpoints = json.load(f)
                logger.debug("Loaded checkpoint metadata containing %d entries.", len(self.top_k_checkpoints))
            except Exception as exc:
                logger.warning("Failed to load checkpoint metadata %s: %s. Re-initializing.", self.meta_path, exc)
                self.top_k_checkpoints = []

    def _save_metadata(self) -> None:
        """Save checkpoint tracking metadata to disk."""
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.top_k_checkpoints, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save checkpoint metadata to %s: %s", self.meta_path, exc)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: TrainerState,
        current_value: float,
    ) -> None:
        """Save a new checkpoint and manage top-K weights.

        Parameters
        ----------
        model : torch.nn.Module
            The model to save.
        optimizer : torch.optim.Optimizer
            The optimizer state to save.
        state : TrainerState
            The current TrainerState.
        current_value : float
            Monitored metric value for the current epoch.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        epoch = state.epoch + 1

        # 1. Check and update best_metric
        is_best = False
        if state.best_metric == -float("inf") or state.best_metric == float("inf"):
            # Initial evaluation
            is_best = True
        elif self.mode == "min" and current_value < state.best_metric:
            is_best = True
        elif self.mode == "max" and current_value > state.best_metric:
            is_best = True

        if is_best:
            state.best_metric = current_value

        # 2. Compile checkpoint payload
        payload = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
            "elapsed_time": state.elapsed_time,
            "learning_rate": state.learning_rate,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        # 3. Save last.pt
        try:
            torch.save(payload, self.last_path)
            logger.debug("Saved last checkpoint to %s", self.last_path)
        except Exception as exc:
            logger.error("Failed to save last checkpoint to %s: %s", self.last_path, exc)

        # 4. Save best.pt if needed
        if is_best:
            try:
                torch.save(payload, self.best_path)
                logger.info(
                    "New best model achieved (value: %.4f). Saved to %s",
                    current_value,
                    self.best_path,
                )
            except Exception as exc:
                logger.error("Failed to save best checkpoint: %s", exc)

        # 4. Save epoch checkpoint and manage top-K
        epoch_filename = f"epoch_{epoch:03d}.pt"
        epoch_path = self.checkpoint_dir / epoch_filename

        try:
            torch.save(payload, epoch_path)
            logger.debug("Saved epoch checkpoint: %s", epoch_path)
        except Exception as exc:
            logger.error("Failed to save epoch checkpoint to %s: %s", epoch_path, exc)
            return

        # Add to top-K tracking
        self.top_k_checkpoints.append({
            "path": str(epoch_path.resolve()),
            "value": current_value,
            "epoch": epoch,
        })

        # Sort top-K list: best value first
        reverse_sort = self.mode == "max"
        self.top_k_checkpoints.sort(key=lambda x: x["value"], reverse=reverse_sort)

        # Prune older/worse checkpoints if over top-K limit
        if len(self.top_k_checkpoints) > self.save_top_k:
            # Pop the worst checkpoint (last item in sorted list)
            worst = self.top_k_checkpoints.pop()
            worst_path = Path(worst["path"])
            if worst_path.is_file() and worst_path.resolve() != self.best_path.resolve() and worst_path.resolve() != self.last_path.resolve():
                try:
                    worst_path.unlink()
                    logger.debug("Pruned older checkpoint from top-K: %s", worst_path.name)
                except Exception as exc:
                    logger.warning("Failed to delete pruned checkpoint %s: %s", worst_path, exc)

        self._save_metadata()

    def resume_training(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str | Path] = None,
    ) -> TrainerState:
        """Load checkpoint payloads to resume a training run.

        If checkpoint_path is None, defaults to resuming from last.pt in checkpoint_dir.

        Parameters
        ----------
        model : torch.nn.Module
            Model instance to load weights into.
        optimizer : torch.optim.Optimizer
            Optimizer instance to restore momentum states into.
        checkpoint_path : str | Path, optional
            Path to specific checkpoint file.

        Returns
        -------
        TrainerState
            The restored TrainerState.
        """
        path = checkpoint_path
        if path is None:
            path = self.last_path

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found to resume: {path}")

        logger.info("Resuming training from checkpoint: %s", path)
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint file {path}: {exc}") from exc

        # Restore weights
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:
            raise RuntimeError(f"State dict mismatch when loading checkpoint: {exc}") from exc

        # Reconstruct TrainerState
        state = TrainerState(
            epoch=checkpoint.get("epoch", 0) + 1,  # Resume at NEXT epoch
            global_step=checkpoint.get("global_step", 0),
            best_metric=checkpoint.get("best_metric", -float("inf")),
            learning_rate=checkpoint.get("learning_rate", 0.0),
            train_loss=checkpoint.get("train_loss", 0.0),
            val_loss=checkpoint.get("val_loss", 0.0),
            elapsed_time=checkpoint.get("elapsed_time", 0.0),
        )

        logger.info(
            "Resumed training state - next_epoch: %d, global_step: %d, best_metric: %.4f",
            state.epoch + 1,
            state.global_step,
            state.best_metric,
        )

        return state
