"""
pathoai/training/run.py
=======================
PathoAI Training Pipeline Runner.

Orchestrates the entire research training workflow: configuration validation,
dataset splits loading, model/loss setup, safety validation verifications,
training/validation cycles, testing, visualization, reporting, and model exports.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pathoai.core.config import ConfigManager
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
from pathoai.training.callbacks.early_stopping import EarlyStopping
from pathoai.training.callbacks.lr_scheduler import LRSchedulerCallback
from pathoai.training.callbacks.model_checkpoint import ModelCheckpoint
from pathoai.training.callbacks.progress import ProgressLogger
from pathoai.training.callbacks.metrics import MetricsCallback
from pathoai.training.checkpoint.manager import CheckpointManager
from pathoai.training.experiment.experiment import Experiment
from pathoai.training.metrics.aggregation import MetricCollection
from pathoai.training.metrics.confusion import ConfusionMatrixMetric
from pathoai.training.metrics.segmentation import SegmentationMetrics
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

logger = get_logger("pathoai.training.run")


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


def run_experiment(config_path: str) -> None:
    """Orchestrate and execute the complete pipeline workflow."""
    start_time = time.time()

    # 1. Load and Validate Configuration
    ConfigManager._config_node = None
    ConfigManager._instance = None
    ConfigManager.initialize(base_config=config_path)
    config = ConfigManager.get_instance()
    validate_training_config(config)

    # 2. Set Random Seeds
    set_global_seed(config.pipeline.seed)

    # 3. Resolve Execution Device
    device_str = config.pipeline.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # 4. Initialize unique experiment folder structure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.pipeline.name}_{timestamp}"
    exp_dir = Path(config.output.base_dir) / experiment_name
    experiment = Experiment(experiment_dir=exp_dir, config=config)
    experiment.setup()

    # Configure logging to write outputs to the experiment logs folder
    configure_logging(
        log_dir=experiment.logs_dir,
        experiment_id=experiment_name,
        level=config.logging.level,
        console=config.logging.get("console", True),
    )

    logger.info(
        "Initializing training experiment pipeline",
        extra={"experiment": experiment_name, "directory": str(exp_dir), "device": str(device)},
    )

    # 5. Load Dataset and apply patient splits (prevent patient data leakage)
    tiger_data = config.data.tiger
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
            patch_size=config.wsi.patch_extraction.patch_size,
            stride=config.wsi.patch_extraction.stride,
            target_mpp=config.wsi.patch_extraction.target_mpp,
            min_tissue_coverage=config.wsi.patch_extraction.min_tissue_coverage,
        )
        logger.info("Applying patient-wise splits allocation...")
        train_entries, val_entries, test_entries = apply_patient_split(
            manifest=manifest_entries,
            train_ratio=config.dataset.train_ratio,
            val_ratio=config.dataset.val_ratio,
            test_ratio=config.dataset.test_ratio,
            seed=config.dataset.split_seed,
        )
        # Save split definitions for reproducibility
        splits_file.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_file, "w") as f:
            json.dump({"train": train_entries, "val": val_entries, "test": test_entries}, f, indent=4)
        logger.info("Saved patient splits file to %s", splits_file)

    def get_avg_coverage(entries: List[Dict[str, Any]]) -> float:
        if not entries:
            return 0.0
        coverages = [e.get("tissue_coverage", 1.0) for e in entries]
        return float(np.mean(coverages))

    dataset_summary = {
        "train": {"n_patches": len(train_entries), "avg_tissue_coverage": get_avg_coverage(train_entries)},
        "val": {"n_patches": len(val_entries), "avg_tissue_coverage": get_avg_coverage(val_entries)},
        "test": {"n_patches": len(test_entries), "avg_tissue_coverage": get_avg_coverage(test_entries)},
    }
    logger.info("Dataset summary partitions loaded successfully.")

    # 6. Create DataLoaders
    train_loader, val_loader, test_loader = get_segmentation_dataloaders(
        train_entries=train_entries,
        val_entries=val_entries,
        test_entries=test_entries,
        config=config,
    )

    # 7. Build Model & Wrap in SegmentationModel
    raw_model = create_model(config)
    wrapped_model = SegmentationModel(raw_model)

    # Log model parameters profile
    generate_model_summary(
        model=wrapped_model,
        output_dir=experiment.experiment_dir,
        input_shape=(1, 3, config.segmentation.input_size, config.segmentation.input_size),
    )

    # 8. Create Loss, Optimizer, and Scheduler
    loss_fn = LossFactory.create_loss(config, class_weights=None)
    optimizer = resolve_optimizer(wrapped_model, config)
    scheduler = resolve_scheduler(optimizer, config)

    # 9. Preflight Safety checks
    run_preflight_verification(
        model=wrapped_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        patch_size=config.segmentation.input_size,
    )

    # 10. Instantiate Callbacks & Logs
    loggers = experiment.get_loggers()

    # Determine monitoring metric configuration
    seg_train = config.segmentation.training
    monitor_metric = seg_train.get("early_stopping_monitor", "val_loss")
    # Determine mode: 'min' for loss, 'max' for dice/iou/accuracy
    monitor_mode = "min" if "loss" in monitor_metric else "max"

    # Setup Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=experiment.checkpoints_dir,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=3,
    )

    # Pack callbacks list
    callbacks = [
        MetricsCallback(n_classes=config.segmentation.n_classes),
        ProgressLogger(),
        model_checkpoint := ModelCheckpoint(checkpoint_manager),
        EarlyStopping(
            monitor=monitor_metric,
            patience=seg_train.get("early_stopping_patience", 15),
            mode=monitor_mode,
        ),
    ]
    if scheduler is not None:
        callbacks.append(LRSchedulerCallback(scheduler, monitor=monitor_metric))

    # Add CSV and TensorBoard loggers
    callbacks.extend(loggers)

    # 11. Create Trainer
    use_amp = config.pipeline.get("mixed_precision", False)
    accumulate_grad = seg_train.get("accumulate_grad_batches", 1)
    grad_clip = seg_train.get("grad_clip_val", None)

    trainer = Trainer(
        model=wrapped_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        state=None,
        callbacks=callbacks,
        use_amp=use_amp,
        accumulate_grad_batches=accumulate_grad,
        grad_clip_val=grad_clip,
    )

    # 12. Run Training fit loop
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=seg_train.epochs,
    )

    # 13. Load Best Model checkpoint weights before testing/exporting
    best_checkpoint_path = checkpoint_manager.best_path
    if best_checkpoint_path is not None and best_checkpoint_path.is_file():
        logger.info("Loading best model weights from %s for testing", best_checkpoint_path)
        wrapped_model.load_weights(best_checkpoint_path)
    else:
        logger.warning("Best checkpoint path not resolved. Proceeding with last training weights.")

    # 14. Testing & Evaluation on partition
    logger.info("Starting final test partition evaluation...")
    test_trainer = Trainer(
        model=wrapped_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    test_trainer.callback_manager.trigger("on_validation_begin", test_trainer)
    test_loss = test_trainer.validate(test_loader)
    test_trainer.callback_manager.trigger("on_validation_end", test_trainer)

    # Retrieve metrics dictionary from the metrics aggregation collection
    test_metrics = {}
    for cb in callbacks:
        if hasattr(cb, "metrics") and isinstance(cb.metrics, MetricCollection):
            # Compute final computed test metrics values
            test_metrics = cb.metrics.compute()
            break

    # Add loss to test metrics
    test_metrics["loss"] = test_loss
    logger.info("Final Test Partition Evaluation Metrics: %s", test_metrics)

    # Save final test metrics to metrics.json
    metrics_path = experiment.metrics_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    # 15. Export model to ONNX & TorchScript
    logger.info("Exporting best model weights to deployment formats...")
    # ONNX export
    try:
        onnx_export_path = experiment.exports_dir / "model.onnx"
        export_model(
            model=wrapped_model,
            output_path=onnx_export_path,
            export_format="onnx",
            input_shape=(1, 3, config.segmentation.input_size, config.segmentation.input_size),
        )
    except Exception as exc:
        logger.warning("Could not export model to ONNX: %s", exc)

    # TorchScript export
    try:
        ts_export_path = experiment.exports_dir / "model.ts"
        export_model(
            model=wrapped_model,
            output_path=ts_export_path,
            export_format="torchscript",
            input_shape=(1, 3, config.segmentation.input_size, config.segmentation.input_size),
        )
    except Exception as exc:
        logger.warning("Could not export model to TorchScript: %s", exc)

    # 16. Plot curves & visualizations if Matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        try:
            logger.info("Generating experiment visualization plots...")
            # Load training log history CSV
            history_csv = experiment.history_dir / "history.csv"
            if history_csv.is_file():
                history_df = pd.read_csv(history_csv)

                # Loss/Metric curves
                curves_plotter = TrainingCurves(output_dir=experiment.curves_dir)
                curves_plotter.plot_history(history_df)

                # Test Confusion matrix
                for cb in callbacks:
                    if hasattr(cb, "metrics"):
                        for m in cb.metrics.metrics:
                            if isinstance(m, ConfusionMatrixMetric):
                                raw_cm = m.confusion_matrix.numpy()
                                class_names = ["Background", "Tumor", "Stroma", "Necrosis", "Other", "Inflammatory"]
                                # Slice names to match config n_classes
                                class_names = class_names[:config.segmentation.n_classes]
                                cm_plotter = ConfusionMatrixPlot(output_dir=experiment.confusion_dir)
                                cm_plotter.plot(raw_cm, class_names=class_names)
                                break

                # Save sample prediction overlays
                inference_engine = SegmentationInference(wrapped_model, device=str(device))
                overlay_plotter = PredictionOverlay(output_dir=experiment.predictions_dir)

                # Get a small batch from test loader
                for images, targets in test_loader:
                    # Run patch predictions
                    for i in range(min(5, len(images))):
                        img_np = images[i].permute(1, 2, 0).numpy()
                        # Unnormalize for visualization
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = (img_np * std + mean) * 255.0
                        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

                        pred_mask = inference_engine.predict_patch(images[i])
                        gt_mask = targets[i].numpy()

                        # Save overlay gallery strip
                        overlay_plotter.plot_gallery(
                            image=img_np,
                            gt_mask=gt_mask,
                            pred_mask=pred_mask,
                            filename=f"test_gallery_{i}.png",
                        )
                    break
        except Exception as exc:
            logger.warning("Failed to generate plots/overlays: %s", exc)
    else:
        logger.warning("Matplotlib is unavailable. Skipping training curves/overlays generation.")

    # 17. Compile Report Generation
    try:
        logger.info("Generating training summary report...")
        report_gen = ReportGenerator(output_dir=experiment.experiment_dir)

        history_csv = experiment.history_dir / "history.csv"
        history_df = pd.read_csv(history_csv) if history_csv.is_file() else pd.DataFrame()

        best_epoch = checkpoint_manager.best_epoch
        # Extract best epoch metrics from history df if present
        best_metrics = {}
        if not history_df.empty and best_epoch is not None:
            # history epochs are usually 0-indexed in CSV, convert from 1-indexed
            epoch_row = history_df[history_df["epoch"] == (best_epoch - 1)]
            if not epoch_row.empty:
                best_metrics = epoch_row.iloc[0].to_dict()

        elapsed_time = time.time() - start_time

        # Save experiment summary file
        report_path = report_gen.generate_report(
            experiment_name=experiment_name,
            config_dict=config.to_dict() if hasattr(config, "to_dict") else {},
            history_df=history_df,
            best_epoch_metrics=best_metrics,
            best_epoch=best_epoch or 0,
            elapsed_time=elapsed_time,
            dataset_summary=dataset_summary,
        )

        # Suffix a symlink or copy to 'experiment_summary.md' in root of experiment dir
        summary_md_dest = experiment.experiment_dir / "experiment_summary.md"
        if report_path.is_file():
            summary_md_dest.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

        logger.info("Saved final experiment summary report to %s", summary_md_dest)
    except Exception as exc:
        logger.error("Failed to generate training experiment report: %s", exc, exc_info=True)

    logger.info(
        "Experiment pipeline finished successfully in %.2f minutes.",
        (time.time() - start_time) / 60.0,
    )


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(description="PathoAI Segmentation Training Pipeline Orchestrator.")
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
        logger.critical("Fatal error in training orchestrator: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
