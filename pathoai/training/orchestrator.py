"""
pathoai/training/orchestrator.py
================================
PathoAI Training Pipeline Orchestrator.

Manages the training workflow state machine, diagnostic verification gates
(pre-epoch-1 safety verification, single-batch overfitting sanity checks),
reproducibility logging, and end-to-end training runs.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.5
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from pathoai.core.config import ConfigNode
from pathoai.core.exceptions import PipelineError, ValidationError
from pathoai.core.logger import configure_logging, get_logger
from pathoai.core.reproducibility import set_global_seed
from pathoai.datasets.loader import get_segmentation_dataloaders
from pathoai.datasets.manifest import generate_dataset_manifest
from pathoai.datasets.split import apply_patient_split
from pathoai.segmentation.export import export_model
from pathoai.segmentation.factory import create_model
from pathoai.segmentation.inference import SegmentationInference
from pathoai.segmentation.losses import LossFactory
from pathoai.segmentation.model import SegmentationModel
from pathoai.segmentation.summary import generate_model_summary
from pathoai.training.callbacks import (
    EarlyStopping,
    LRSchedulerCallback,
    MetricsCallback,
    ModelCheckpoint,
    ProgressLogger,
)
from pathoai.training.checkpoint.manager import CheckpointManager
from pathoai.training.experiment.experiment import Experiment
from pathoai.training.metrics.confusion import ConfusionMatrixMetric
from pathoai.training.metrics.aggregation import MetricCollection
from pathoai.training.reports.report_generator import ReportGenerator
from pathoai.training.trainer.trainer import Trainer

# Optional visualization imports
try:
    from pathoai.training.visualization.confusion import ConfusionMatrixPlot
    from pathoai.training.visualization.curves import TrainingCurves
    from pathoai.training.visualization.overlays import PredictionOverlay
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = get_logger("pathoai.training.orchestrator")


def validate_training_config(config: Any) -> None:
    """Perform pre-flight sanity checks on config settings."""
    if not hasattr(config, "segmentation"):
        raise ValidationError("Missing 'segmentation' section in configuration file.")
    if not hasattr(config, "pipeline"):
        raise ValidationError("Missing 'pipeline' section in configuration file.")
    if not hasattr(config, "data") or not hasattr(config.data, "tiger"):
        raise ValidationError("Missing 'data.tiger' section in configuration file.")

    # Validate segmentation training settings
    seg_cfg = config.segmentation
    if not hasattr(seg_cfg, "training"):
        raise ValidationError("Missing 'segmentation.training' settings.")


def resolve_optimizer(model: nn.Module, config: Any) -> torch.optim.Optimizer:
    """Instantiate the PyTorch optimizer based on configuration settings."""
    seg_train = config.segmentation.training
    opt_name = seg_train.get("optimizer_name", "adamw").lower()
    lr = seg_train.get("learning_rate", 1e-4)
    wd = seg_train.get("weight_decay", 1e-4)

    logger.debug("Resolving optimizer", extra={"optimizer": opt_name, "lr": lr, "weight_decay": wd})

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        momentum = seg_train.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        raise ValidationError(f"Unsupported optimizer name: '{opt_name}'")


def resolve_scheduler(optimizer: torch.optim.Optimizer, config: Any) -> Optional[Any]:
    """Instantiate the learning rate scheduler based on configuration settings."""
    seg_train = config.segmentation.training
    sched_name = seg_train.get("lr_scheduler", "cosine").lower()

    logger.debug("Resolving scheduler", extra={"scheduler": sched_name})

    if sched_name == "cosine":
        epochs = seg_train.get("epochs", 50)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "plateau":
        patience = seg_train.get("scheduler_patience", 5)
        # mode: 'min' for loss, 'max' for metrics
        monitor = seg_train.get("early_stopping_monitor", "val_loss")
        mode = "min" if "loss" in monitor else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience)
    elif sched_name == "none" or not sched_name:
        return None
    else:
        raise ValidationError(f"Unsupported scheduler name: '{sched_name}'")


def run_preflight_verification(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    patch_size: int = 256,
) -> None:
    """M5.19 Pre-Epoch-1 Safety verification.

    Runs a full dummy iteration (forward, backward, optimizer step, metric update)
    to confirm correctness and abort immediately if any pipeline components fail.
    """
    logger.info("Executing Pre-Epoch-1 Safety verification...")
    model.train()
    model.to(device)

    # 1. Dummy Forward Pass
    try:
        dummy_in = torch.zeros(2, 3, patch_size, patch_size, dtype=torch.float32, device=device)
        dummy_tgt = torch.zeros(2, patch_size, patch_size, dtype=torch.long, device=device)
        optimizer.zero_grad()
        out = model(dummy_in)
    except Exception as exc:
        raise PipelineError(f"Pre-flight verification failed during forward pass: {exc}") from exc

    # 2. Loss Computation
    try:
        loss = loss_fn(out, dummy_tgt)
    except Exception as exc:
        raise PipelineError(f"Pre-flight verification failed during loss calculation: {exc}") from exc

    # 3. Backward Pass & Step
    try:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    except Exception as exc:
        raise PipelineError(f"Pre-flight verification failed during backward/optimizer step: {exc}") from exc

    # 4. Metric Computation
    try:
        mc = MetricCollection(n_classes=out.shape[1])
        mc.update(out.detach().cpu(), dummy_tgt.cpu())
        mc.compute()
        mc.reset()
    except Exception as exc:
        raise PipelineError(f"Pre-flight verification failed during metrics accumulation: {exc}") from exc

    logger.info("Pre-Epoch-1 Safety verification completed successfully.")


class TrainingOrchestrator:
    """Orchestrates the lifecycle of a semantic segmentation training experiment.

    Maintains a strict state machine to log exactly where the pipeline is
    operating or if it crashed, and executes validation safety gates before training.
    """

    def __init__(self, config: ConfigNode, config_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        config : ConfigNode
            Loaded type-safe configuration singleton node.
        config_path : str, optional
            Path to the config file on disk (for SHA hashing).
        """
        self.config = config
        self.config_path = config_path
        self.device = torch.device("cpu")
        self.experiment: Optional[Experiment] = None
        self.experiment_name = ""
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.wrapped_model: Optional[SegmentationModel] = None
        self.loss_fn: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.callbacks: List[Any] = []
        self.dataset_summary: Dict[str, Any] = {}
        self.start_time = 0.0

        self.state = "INITIALIZING"
        self._set_state("INITIALIZING")

    def _set_state(self, new_state: str) -> None:
        """Transition the pipeline to a new state, logging and writing state to disk."""
        old_state = self.state
        self.state = new_state
        logger.info("Pipeline state transition: %s -> %s", old_state, new_state)
        
        if self.experiment is not None:
            state_file = self.experiment.experiment_dir / "experiment_state.json"
            try:
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "state": self.state,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "last_transition_from": old_state
                    }, f, indent=4)
            except Exception as exc:
                logger.warning("Failed to save experiment state file: %s", exc)

    def run(self) -> None:
        """Executes the complete orchestration workflow lifecycle."""
        self.start_time = time.time()
        
        # 1. Setup random seeds
        set_global_seed(self.config.pipeline.seed)

        # 2. Resolve Execution Device
        device_str = self.config.pipeline.device
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # 3. Setup Experiment Directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.config.pipeline.name}_{timestamp}"
        exp_dir = Path(self.config.output.base_dir) / self.experiment_name
        self.experiment = Experiment(experiment_dir=exp_dir, config=self.config)
        self.experiment.setup()

        # Update state to READY in the directory
        self._set_state("READY")

        # Configure logs folder logging
        configure_logging(
            log_dir=self.experiment.logs_dir,
            experiment_id=self.experiment_name,
            level=self.config.logging.level,
            console=self.config.logging.get("console", True),
        )

        logger.info(
            "Initializing training experiment pipeline",
            extra={"experiment": self.experiment_name, "directory": str(exp_dir), "device": str(self.device)},
        )

        # 4. Prepare Dataset splits
        self._prepare_datasets()

        # 5. Build Model architecture & Wrapper
        raw_model = create_model(self.config)
        self.wrapped_model = SegmentationModel(raw_model)

        # Generate structural documentation report
        generate_model_summary(
            model=self.wrapped_model,
            output_dir=self.experiment.experiment_dir,
            input_shape=(1, 3, self.config.segmentation.input_size, self.config.segmentation.input_size),
        )

        # 6. Resolve optimizer, scheduler, and loss function
        self.loss_fn = LossFactory.create_loss(self.config, class_weights=None)
        self.optimizer = self._resolve_optimizer()
        self.scheduler = self._resolve_scheduler()

        # 7. Setup Callbacks & Checkpoint Manager
        self._setup_callbacks()

        # 8. Execute extended preflight verification and overfit diagnostics
        self._run_preflight_and_sanity()

        # 9. Fit Loop execution
        self._set_state("TRAINING")
        trainer = Trainer(
            model=self.wrapped_model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            state=None,
            callbacks=self.callbacks,
            use_amp=self.config.pipeline.get("mixed_precision", False),
            accumulate_grad_batches=self.config.segmentation.training.get("accumulate_grad_batches", 1),
            grad_clip_val=self.config.segmentation.training.get("grad_clip_val", None),
        )

        trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.config.segmentation.training.epochs,
        )

        # 10. testing evaluation
        self._set_state("TESTING")
        best_checkpoint_path = self.checkpoint_manager.best_path
        if best_checkpoint_path is not None and best_checkpoint_path.is_file():
            logger.info("Loading best model weights from %s for testing", best_checkpoint_path)
            self.wrapped_model.load_weights(best_checkpoint_path)
        else:
            logger.warning("Best checkpoint path not resolved. Proceeding with last training weights.")

        test_trainer = Trainer(
            model=self.wrapped_model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            callbacks=self.callbacks,
        )
        # Execute test end callbacks
        test_trainer.callback_manager.trigger("on_validation_begin", test_trainer)
        test_loss = test_trainer.validate(self.test_loader)
        test_trainer.callback_manager.trigger("on_validation_end", test_trainer)

        test_metrics = {}
        for cb in self.callbacks:
            if hasattr(cb, "metrics") and isinstance(cb.metrics, MetricCollection):
                test_metrics = cb.metrics.compute()
                break
        test_metrics["loss"] = test_loss
        logger.info("Final Test Partition Evaluation Metrics: %s", test_metrics)

        # Save metrics to json file
        metrics_path = self.experiment.metrics_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=4)

        # 11. Model exports compilation
        self._set_state("EXPORTING")
        self._export_models()

        # 12. Visualization curves
        if MATPLOTLIB_AVAILABLE:
            self._generate_visualizations(test_metrics)

        # 13. Reports generation
        self._compile_final_report(test_metrics)

        self._set_state("FINISHED")

    def _prepare_datasets(self) -> None:
        """Loads or generates dataset splits and creates PyTorch DataLoaders."""
        tiger_data = self.config.data.tiger
        splits_file = Path(tiger_data.splits_file)

        if splits_file.is_file():
            logger.info("Loading existing patient splits file from %s", splits_file)
            with open(splits_file, "r") as f:
                splits = json.load(f)
            train_entries = splits["train"]
            val_entries = splits["val"]
            test_entries = splits["test"]
        else:
            logger.info("Splits file not found. Generating manifest from %s", tiger_data.train_dir)
            manifest_entries = generate_dataset_manifest(
                dataset_root=tiger_data.train_dir,
                patch_size=self.config.wsi.patch_extraction.patch_size,
                stride=self.config.wsi.patch_extraction.stride,
                target_mpp=self.config.wsi.patch_extraction.target_mpp,
                min_tissue_coverage=self.config.wsi.patch_extraction.min_tissue_coverage,
            )
            logger.info("Applying patient-wise splits allocation...")
            train_entries, val_entries, test_entries = apply_patient_split(
                manifest=manifest_entries,
                train_ratio=self.config.dataset.train_ratio,
                val_ratio=self.config.dataset.val_ratio,
                test_ratio=self.config.dataset.test_ratio,
                seed=self.config.dataset.split_seed,
            )
            splits_file.parent.mkdir(parents=True, exist_ok=True)
            with open(splits_file, "w") as f:
                json.dump({"train": train_entries, "val": val_entries, "test": test_entries}, f, indent=4)
            logger.info("Saved patient splits file to %s", splits_file)

        def get_avg_coverage(entries: List[Dict[str, Any]]) -> float:
            if not entries:
                return 0.0
            coverages = [e.get("tissue_coverage", 1.0) for e in entries]
            return float(np.mean(coverages))

        self.dataset_summary = {
            "train": {"n_patches": len(train_entries), "avg_tissue_coverage": get_avg_coverage(train_entries)},
            "val": {"n_patches": len(val_entries), "avg_tissue_coverage": get_avg_coverage(val_entries)},
            "test": {"n_patches": len(test_entries), "avg_tissue_coverage": get_avg_coverage(test_entries)},
        }
        logger.info("Dataset summary partitions loaded successfully.")

        loaders = get_segmentation_dataloaders(
            train_entries=train_entries,
            val_entries=val_entries,
            test_entries=test_entries,
            config=self.config,
        )
        self.train_loader, self.val_loader, self.test_loader = loaders

    def _resolve_optimizer(self) -> torch.optim.Optimizer:
        """Instantiate the optimizer using module-level resolver."""
        return resolve_optimizer(self.wrapped_model, self.config)

    def _resolve_scheduler(self) -> Optional[Any]:
        """Instantiate the scheduler using module-level resolver."""
        return resolve_scheduler(self.optimizer, self.config)

    def _setup_callbacks(self) -> None:
        """Configures checkpoints callbacks, early stopping, schedulers, and loggers."""
        seg_train = self.config.segmentation.training
        monitor_metric = seg_train.get("early_stopping_monitor", "val_loss")
        monitor_mode = "min" if "loss" in monitor_metric else "max"

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.experiment.checkpoints_dir,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=3,
        )

        self.callbacks = [
            MetricsCallback(n_classes=self.config.segmentation.n_classes),
            ProgressLogger(),
            ModelCheckpoint(self.checkpoint_manager),
            EarlyStopping(
                monitor=monitor_metric,
                patience=seg_train.get("early_stopping_patience", 15),
                mode=monitor_mode,
            ),
        ]

        if self.scheduler is not None:
            self.callbacks.append(LRSchedulerCallback(self.scheduler, monitor=monitor_metric))

        loggers = self.experiment.get_loggers()
        self.callbacks.extend(loggers)

    def _run_preflight_and_sanity(self) -> None:
        """Executes extended startup safety verifications and overfit gates."""
        logger.info("Executing extended preflight pipeline validations...")
        patch_size = self.config.segmentation.input_size

        # 1-3. Core preflight forward/backward/metric checks
        run_preflight_verification(
            model=self.wrapped_model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            patch_size=patch_size,
        )

        # 4. Checkpoint save check
        try:
            dummy_state_file = self.experiment.checkpoints_dir / "preflight_dummy.pt"
            torch.save({"dummy": True}, dummy_state_file)
            dummy_state_file.unlink()
        except Exception as exc:
            raise PipelineError(f"Preflight check failed (checkpoint write/delete permission): {exc}") from exc

        # 5. Exporter run check
        try:
            dummy_ts_file = self.experiment.exports_dir / "preflight_dummy.ts"
            export_model(self.wrapped_model, dummy_ts_file, export_format="torchscript", input_shape=(1, 3, patch_size, patch_size))
            dummy_ts_file.unlink()
        except Exception as exc:
            raise PipelineError(f"Preflight check failed (TorchScript tracing compiler): {exc}") from exc

        # 6. Report compilation check
        try:
            rg = ReportGenerator(output_dir=self.experiment.experiment_dir)
            dummy_report_path = rg.generate_report(
                experiment_name="preflight_dummy",
                config_dict={},
                history_df=pd.DataFrame(),
                best_epoch_metrics={},
                best_epoch=1,
                elapsed_time=1.0,
                dataset_summary={},
            )
            if dummy_report_path.is_file():
                dummy_report_path.unlink()
        except Exception as exc:
            raise PipelineError(f"Preflight check failed (Markdown report generation): {exc}") from exc

        logger.info("Preflight safety verifications completed successfully.")

        # 7. Single-batch overfit sanity check
        if self.config.pipeline.get("sanity_overfit", True) and self.train_loader is not None:
            logger.info("Executing diagnostic single-batch overfit sanity check...")
            self.wrapped_model.train()
            
            # Fetch first batch
            batch_iter = iter(self.train_loader)
            images, targets = next(batch_iter)
            images = images.to(self.device)
            targets = targets.to(self.device)

            losses = []
            for step in range(50):
                self.optimizer.zero_grad()
                outputs = self.wrapped_model(images)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            initial_loss = losses[0]
            final_loss = losses[-1]
            logger.info("Diagnostic Overfit - Initial Loss: %.4f -> Final Loss (step 50): %.4f", initial_loss, final_loss)

            if final_loss > (initial_loss * 0.5):
                logger.warning("Sanity check warning: Model did not overfit effectively (loss stayed above 50%% of initial value). "
                               "Check labels, learning rates, or model layers.")
            else:
                logger.info("Diagnostic overfit check passed successfully.")

            # Reset optimizer and model parameters to prevent weight leakage
            self.optimizer.zero_grad()
            for layer in self.wrapped_model.modules():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def _export_models(self) -> None:
        """Trace and compile final model weights to ONNX and TorchScript."""
        logger.info("Exporting best model weights to deployment formats...")
        input_size = self.config.segmentation.input_size
        
        # ONNX export
        try:
            onnx_export_path = self.experiment.exports_dir / "model.onnx"
            export_model(
                model=self.wrapped_model,
                output_path=onnx_export_path,
                export_format="onnx",
                input_shape=(1, 3, input_size, input_size),
            )
        except Exception as exc:
            logger.warning("Could not export model to ONNX: %s", exc)

        # TorchScript export
        try:
            ts_export_path = self.experiment.exports_dir / "model.ts"
            export_model(
                model=self.wrapped_model,
                output_path=ts_export_path,
                export_format="torchscript",
                input_shape=(1, 3, input_size, input_size),
            )
        except Exception as exc:
            logger.warning("Could not export model to TorchScript: %s", exc)

    def _generate_visualizations(self, test_metrics: Dict[str, Any]) -> None:
        """Generates matplotlib curves and sample segmentation overlay galleries."""
        try:
            logger.info("Generating experiment visualization plots...")
            history_csv = self.experiment.history_dir / "history.csv"
            if history_csv.is_file():
                history_df = pd.read_csv(history_csv)

                # Loss/Metric curves
                curves_plotter = TrainingCurves(output_dir=self.experiment.curves_dir)
                curves_plotter.plot_history(history_df)

                # Test Confusion matrix
                for cb in self.callbacks:
                    if hasattr(cb, "metrics"):
                        for m in cb.metrics.metrics:
                            if isinstance(m, ConfusionMatrixMetric):
                                raw_cm = m.confusion_matrix.numpy()
                                class_names = ["Background", "Tumor", "Stroma", "Necrosis", "Other", "Inflammatory"]
                                class_names = class_names[:self.config.segmentation.n_classes]
                                cm_plotter = ConfusionMatrixPlot(output_dir=self.experiment.confusion_dir)
                                cm_plotter.plot(raw_cm, class_names=class_names)
                                break

                # Save sample prediction overlays
                inference_engine = SegmentationInference(self.wrapped_model, device=str(self.device))
                overlay_plotter = PredictionOverlay(output_dir=self.experiment.predictions_dir)

                for images, targets in self.test_loader:
                    for i in range(min(5, len(images))):
                        img_np = images[i].permute(1, 2, 0).numpy()
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = (img_np * std + mean) * 255.0
                        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

                        pred_mask = inference_engine.predict_patch(images[i])
                        gt_mask = targets[i].numpy()

                        overlay_plotter.plot_gallery(
                            image=img_np,
                            gt_mask=gt_mask,
                            pred_mask=pred_mask,
                            filename=f"test_gallery_{i}.png",
                        )
                    break
        except Exception as exc:
            logger.warning("Failed to generate plots/overlays: %s", exc)

    def _compile_final_report(self, test_metrics: Dict[str, Any]) -> None:
        """Gathers extensive reproducibility metrics and writes the final markdown report."""
        try:
            logger.info("Generating training summary report...")
            report_gen = ReportGenerator(output_dir=self.experiment.experiment_dir)

            history_csv = self.experiment.history_dir / "history.csv"
            history_df = pd.read_csv(history_csv) if history_csv.is_file() else pd.DataFrame()

            best_epoch = self.checkpoint_manager.best_epoch
            best_metrics = {}
            if not history_df.empty and best_epoch is not None:
                epoch_row = history_df[history_df["epoch"] == (best_epoch - 1)]
                if not epoch_row.empty:
                    best_metrics = epoch_row.iloc[0].to_dict()

            elapsed_time = time.time() - self.start_time

            # Gather extended system reproducibility specifications
            git_hash = "N/A"
            try:
                git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            except Exception:
                pass

            config_hash = "N/A"
            if self.config_path is not None:
                try:
                    config_hash = self._get_file_hash(Path(self.config_path))
                except Exception:
                    pass

            splits_hash = "N/A"
            tiger_data = self.config.data.tiger
            splits_path = Path(tiger_data.splits_file)
            if splits_path.is_file():
                try:
                    splits_hash = self._get_file_hash(splits_path)
                except Exception:
                    pass

            gpu_name = "CPU Only"
            gpu_driver = "N/A"
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                # Attempt to get driver version via nvidia-smi if on windows
                try:
                    smi_out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode("utf-8").strip()
                    gpu_driver = smi_out
                except Exception:
                    pass

            reproducibility_meta = {
                "Git Hash": git_hash,
                "Config File Hash": config_hash,
                "Dataset Splits File Hash": splits_hash,
                "Hostname": socket.gethostname(),
                "GPU Name": gpu_name,
                "GPU Driver Version": gpu_driver,
                "Python Version": sys.version.split(" ")[0],
                "PyTorch Version": torch.__version__,
                "CUDA Version": torch.version.cuda or "N/A",
            }

            # Save reproducibility definitions to final stats file
            meta_dest = self.experiment.experiment_dir / "reproducibility_metadata.json"
            with open(meta_dest, "w", encoding="utf-8") as f:
                json.dump(reproducibility_meta, f, indent=4)

            # Generate and copy markdown summary report
            report_path = report_gen.generate_report(
                experiment_name=self.experiment_name,
                config_dict=self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                history_df=history_df,
                best_epoch_metrics=best_metrics,
                best_epoch=best_epoch or 0,
                elapsed_time=elapsed_time,
                dataset_summary=self.dataset_summary,
            )

            # Append reproducibility metrics to the end of the markdown report
            if report_path.is_file():
                content = report_path.read_text(encoding="utf-8")
                
                meta_section = [
                    "\n## 🔬 Detailed Provenance & Reproducibility Metrics\n",
                    f"- **Git Commit Hash:** `{reproducibility_meta['Git Hash']}`",
                    f"- **Config Hash:** `{reproducibility_meta['Config File Hash']}`",
                    f"- **Dataset Splits Hash:** `{reproducibility_meta['Dataset Splits File Hash']}`",
                    f"- **Execution Hostname:** `{reproducibility_meta['Hostname']}`",
                    f"- **GPU Model:** `{reproducibility_meta['GPU Name']}`",
                    f"- **GPU Driver:** `{reproducibility_meta['GPU Driver Version']}`",
                    f"- **CUDA Version:** `{reproducibility_meta['CUDA Version']}`",
                    f"- **PyTorch Version:** `{reproducibility_meta['PyTorch Version']}`",
                    f"- **Python Version:** `{reproducibility_meta['Python Version']}`\n",
                ]
                
                full_content = content + "\n".join(meta_section)
                report_path.write_text(full_content, encoding="utf-8")

                # Copy report file to experiment root folder
                summary_md_dest = self.experiment.experiment_dir / "experiment_summary.md"
                summary_md_dest.write_text(full_content, encoding="utf-8")
                logger.info("Saved final experiment summary report to %s", summary_md_dest)

        except Exception as exc:
            logger.error("Failed to generate training experiment report: %s", exc, exc_info=True)

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate the SHA256 checksum hash of a target file."""
        import hashlib
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
