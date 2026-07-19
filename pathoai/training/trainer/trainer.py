"""
pathoai/training/trainer/trainer.py
==================================
Model-agnostic training loop engine.

Encapsulates training, validation, testing, and prediction loops. Interacts
with models, losses, and optimizers using dependency injection, notifying
registered callbacks at key stages of the lifecycle.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.1
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from pathoai.core.exceptions import PipelineError
from pathoai.core.logger import get_logger
from pathoai.training.trainer.state import TrainerState

logger = get_logger(__name__)


class CallbackManager:
    """Orchestrates callback execution order and routes event hooks."""

    def __init__(self, callbacks: Optional[List[Any]] = None) -> None:
        self.callbacks = callbacks or []

    def trigger(self, event_name: str, trainer: Trainer, *args: Any, **kwargs: Any) -> None:
        """Call the specified event hook on all registered callbacks in order."""
        for callback in self.callbacks:
            hook = getattr(callback, event_name, None)
            if hook and callable(hook):
                try:
                    hook(trainer, *args, **kwargs)
                except Exception as exc:
                    logger.error("Error in callback %s during %s: %s", callback.__class__.__name__, event_name, exc)
                    raise PipelineError(f"Callback {callback.__class__.__name__} failed: {exc}") from exc


class Trainer:
    """Model-agnostic trainer that encapsulates PyTorch training loops."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: Union[str, torch.device] = "cpu",
        state: Optional[TrainerState] = None,
        callbacks: Optional[List[Any]] = None,
        use_amp: bool = False,
        accumulate_grad_batches: int = 1,
        grad_clip_val: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to train.
        optimizer : torch.optim.Optimizer
            The optimizer used for training.
        loss_fn : torch.nn.Module
            The loss function.
        device : str | torch.device
            The execution device ('cpu' or 'cuda').
        state : TrainerState, optional
            TrainerState object to restore state or initialize a new one.
        callbacks : List[Callback], optional
            List of Callback observers to monitor training.
        use_amp : bool
            Whether to use Automatic Mixed Precision.
        accumulate_grad_batches : int
            Number of batches to accumulate gradients over before optimizer step.
        grad_clip_val : float, optional
            Maximum gradient norm value for clipping.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.state = state or TrainerState()

        # Callbacks manager
        self.callback_manager = CallbackManager(callbacks)

        # Control flags
        self.stop_training = False

        # Advanced training params
        self.use_amp = use_amp and self.device.type == "cuda"
        self.accumulate_grad_batches = max(1, accumulate_grad_batches)
        self.grad_clip_val = grad_clip_val

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Holders for batch/epoch intermediates (can be accessed by metrics/callbacks)
        self.current_batch_img: Optional[torch.Tensor] = None
        self.current_batch_lbl: Optional[torch.Tensor] = None
        self.current_batch_pred: Optional[torch.Tensor] = None
        self.current_batch_loss: Optional[torch.Tensor] = None

        # Holds accumulated epoch outputs for evaluation hooks (M4.11)
        self.epoch_preds: List[torch.Tensor] = []
        self.epoch_targets: List[torch.Tensor] = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> TrainerState:
        """Execute the training and validation loop.

        Parameters
        ----------
        train_loader : DataLoader
            Loader for the training data.
        val_loader : DataLoader, optional
            Loader for validation data.
        epochs : int
            Number of epochs to train.

        Returns
        -------
        TrainerState
            Final training state.
        """
        self.stop_training = False
        self.callback_manager.trigger("on_train_begin", self)

        logger.info(
            "Starting training run",
            extra={
                "epochs": epochs,
                "device": str(self.device),
                "use_amp": self.use_amp,
                "accumulation_batches": self.accumulate_grad_batches,
                "grad_clip": self.grad_clip_val,
            },
        )

        for epoch in range(self.state.epoch, epochs):
            if self.stop_training:
                logger.info("Training early stop signal detected.")
                break

            self.state.epoch = epoch
            self.callback_manager.trigger("on_epoch_begin", self)

            epoch_start_time = time.time()

            # 1. Training epoch loop
            train_loss = self.train_epoch(train_loader)
            self.state.train_loss = train_loss

            # 2. Validation loop (if loader provided)
            val_loss = 0.0
            if val_loader is not None:
                self.callback_manager.trigger("on_validation_begin", self)
                val_loss = self.validate(val_loader)
                self.state.val_loss = val_loss
                self.callback_manager.trigger("on_validation_end", self)

            self.state.elapsed_time += time.time() - epoch_start_time
            # Learning rate check
            for param_group in self.optimizer.param_groups:
                self.state.learning_rate = param_group["lr"]
                break

            self.callback_manager.trigger("on_epoch_end", self)

        self.callback_manager.trigger("on_train_end", self)
        logger.info("Training run completed", extra={"total_epochs": self.state.epoch + 1})
        return self.state

    def train_epoch(self, loader: DataLoader) -> float:
        """Run a single epoch of training with AMP, clipping, and accumulation.

        Parameters
        ----------
        loader : DataLoader
            Dataloader containing training data.

        Returns
        -------
        float
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)
        self.num_batches = n_batches

        self.optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(loader):
            self.batch_idx = batch_idx
            self.callback_manager.trigger("on_batch_begin", self)

            images = images.to(self.device)
            targets = targets.to(self.device)

            self.current_batch_img = images
            self.current_batch_lbl = targets

            # Forward pass under autocast if AMP is active
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, targets)
                    # Scale loss to adjust for gradient accumulation
                    loss = loss / self.accumulate_grad_batches
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                loss = loss / self.accumulate_grad_batches

            self.current_batch_pred = outputs
            self.current_batch_loss = loss

            # Check loss validity
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss value is NaN or Inf at epoch {self.state.epoch + 1}, "
                    f"batch {batch_idx + 1}: {loss.item()}"
                )

            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer Step & Clip (only at accumulation boundaries)
            is_update_step = (batch_idx + 1) % self.accumulate_grad_batches == 0 or (batch_idx + 1) == n_batches
            if is_update_step:
                if self.use_amp and self.scaler is not None:
                    # Unscale gradients before clipping
                    if self.grad_clip_val is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulate_grad_batches
            self.state.global_step += 1

            self.callback_manager.trigger("on_batch_end", self)

        # Clear batch intermediates
        self.current_batch_img = None
        self.current_batch_lbl = None
        self.current_batch_pred = None
        self.current_batch_loss = None
        self.batch_idx = None
        self.num_batches = None

        return total_loss / n_batches if n_batches > 0 else 0.0

    def validate(self, loader: DataLoader) -> float:
        """Run validation loop, accumulating predictions for metric hooks.

        Parameters
        ----------
        loader : DataLoader
            Validation dataloader.

        Returns
        -------
        float
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = len(loader)
        self.num_batches = n_batches

        self.epoch_preds.clear()
        self.epoch_targets.clear()

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(loader):
                self.batch_idx = batch_idx
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

                # Accumulate for metrics evaluation (M4.11 hooks)
                # Keep them on CPU to prevent GPU RAM bloat during validation
                self.epoch_preds.append(outputs.detach().cpu())
                self.epoch_targets.append(targets.detach().cpu())

        self.batch_idx = None
        self.num_batches = None
        return total_loss / n_batches if n_batches > 0 else 0.0

    def test(self, loader: DataLoader) -> float:
        """Run testing evaluation (equivalent to validation run)."""
        logger.info("Starting test evaluation")
        test_loss = self.validate(loader)
        return test_loss

    def predict(self, loader: DataLoader) -> List[torch.Tensor]:
        """Run inference over a loader, returning all prediction outputs.

        Parameters
        ----------
        loader : DataLoader
            Prediction dataloader.

        Returns
        -------
        List[torch.Tensor]
            List of model output prediction tensors.
        """
        self.model.eval()
        predictions: List[torch.Tensor] = []

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions.append(outputs.detach().cpu())

        return predictions
