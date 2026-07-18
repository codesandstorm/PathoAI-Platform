"""
pathoai/training/reports/report_generator.py
============================================
Training Experiment Report Generator.

Automatically creates a research-grade Markdown report summarizing system environment,
dataset splits, final and best epoch metrics, loss curves, confusion matrices,
and prediction overlays.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.9
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates comprehensive training experiment Markdown reports."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def _get_git_commit(self) -> str:
        """Safely fetch current Git commit hash."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return "Unknown / Non-git repository"

    def _get_hardware_summary(self) -> Dict[str, str]:
        """Compile a summary of available system hardware resources."""
        summary = {
            "OS": f"{platform.system()} {platform.release()} (v{platform.version()})",
            "Python": platform.python_version(),
            "CPU": platform.processor() or "Unknown CPU",
            "PyTorch": torch.__version__,
        }
        if torch.cuda.is_available():
            summary["GPU"] = f"{torch.cuda.get_device_name(0)} (CUDA Available)"
        else:
            summary["GPU"] = "CPU Only (CUDA Unavailable)"
        return summary

    def generate_report(
        self,
        experiment_name: str,
        config_dict: Dict[str, Any],
        history_df: Any,  # pandas DataFrame
        best_epoch_metrics: Dict[str, Any],
        best_epoch: int,
        elapsed_time: float,
        dataset_summary: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Generate and save the report.md markdown file.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        config_dict : Dict[str, Any]
            Pipeline configuration settings dictionary.
        history_df : pd.DataFrame
            DataFrame of training history.
        best_epoch_metrics : Dict[str, Any]
            Validation metrics dict for the best epoch.
        best_epoch : int
            1-indexed best epoch count.
        elapsed_time : float
            Total training elapsed time in seconds.
        dataset_summary : Dict[str, Any], optional
            Summary profile of training/validation/test partitions.

        Returns
        -------
        Path
            Path to the written report.md file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "report.md"

        git_hash = self._get_git_commit()
        hw = self._get_hardware_summary()

        # Build markdown text
        md = []
        md.append(f"# Experiment Training Report — {experiment_name}\n")

        # 1. Executive Summary
        md.append("## 📊 Executive Summary\n")
        md.append(f"- **Total Training Duration:** {elapsed_time / 60.0:.2f} minutes")
        md.append(f"- **Best Epoch:** Epoch {best_epoch}")

        # Metrics values check
        monitor_val = best_epoch_metrics.get("mean_dice", 0.0)
        md.append(f"- **Best Mean Dice:** {monitor_val:.4f}")
        val_iou = best_epoch_metrics.get("mean_iou", 0.0)
        md.append(f"- **Best Mean IoU (mIoU):** {val_iou:.4f}")
        val_f1 = best_epoch_metrics.get("macro_f1", 0.0)
        md.append(f"- **Best Macro F1:** {val_f1:.4f}")
        md.append(f"- **Best Pixel Accuracy:** {best_epoch_metrics.get('pixel_accuracy', 0.0):.4f}\n")

        # 2. System and Reproducibility Metadata
        md.append("## ⚙️ Environment & Reproducibility\n")
        md.append(f"- **Git Commit Hash:** `{git_hash}`")
        md.append(f"- **Execution Device:** `{hw['GPU']}`")
        md.append(f"- **CPU Platform:** `{hw['CPU']}`")
        md.append(f"- **OS Version:** `{hw['OS']}`")
        md.append(f"- **PyTorch Version:** `{hw['PyTorch']}`")
        md.append(f"- **Python Version:** `{hw['Python']}`\n")

        # 3. Dataset Configuration
        if dataset_summary:
            md.append("## 📂 Dataset Profile\n")
            md.append("| Split | Number of Patches | Avg. Tissue Coverage |")
            md.append("|---|---|---|")
            for split_name, profile in dataset_summary.items():
                n_patches = profile.get("n_patches", 0)
                cov = profile.get("avg_tissue_coverage", 0.0)
                md.append(f"| {split_name.capitalize()} | {n_patches} | {cov:.4f} |")
            md.append("\n")

        # 4. Class-specific Metrics at Best Epoch
        md.append("## 🎯 Class-specific Performance (Best Epoch)\n")
        dice_per_class = best_epoch_metrics.get("dice_per_class", [])
        iou_per_class = best_epoch_metrics.get("iou_per_class", [])
        support_per_class = best_epoch_metrics.get("support_per_class", [])

        if dice_per_class and iou_per_class:
            md.append("| Class ID | Class Name | Dice Score | IoU Score | Support (pixels) |")
            md.append("|---|---|---|---|---|")
            # TIGER Class mappings: 0=background, 1=invasive tumor, 2=tumor-associated stroma,
            # 3=in-situ tumor, 4=healthy glands, 5=necrosis
            class_names = [
                "Background",
                "Invasive Tumor",
                "Tumor Stroma",
                "In-situ Tumor",
                "Healthy Glands",
                "Necrosis",
            ]
            for c in range(min(len(dice_per_class), len(class_names))):
                support = support_per_class[c] if c < len(support_per_class) else 0
                md.append(f"| {c} | {class_names[c]} | {dice_per_class[c]:.4f} | {iou_per_class[c]:.4f} | {support:,} |")
            md.append("\n")

        # 5. Learning Curves (Embedded reference paths)
        md.append("## 📈 Learning Curves\n")
        md.append("Below are the metrics and loss curves generated for this run:\n")
        md.append("![Loss Curve](../curves/loss.png)\n")
        md.append("![Dice Curve](../curves/dice.png)\n")
        md.append("![mIoU Curve](../curves/iou.png)\n")

        # 6. Confusion Matrix
        md.append("## 🔀 Confusion Matrix\n")
        md.append("Normalized confusion matrix heatmap detailing misclassifications:\n")
        md.append("![Normalized Confusion Matrix](../confusion/normalized_confusion.png)\n")

        # 7. Predictions Gallery Sample
        md.append("## 🖼️ Prediction Visualizations\n")
        md.append("Example overlay prediction showing segmentations and difference maps:\n")
        md.append("![Prediction Sample](../predictions/epoch_best/sample_best.png)\n")

        # Write to file
        try:
            report_path.write_text("\n".join(md), encoding="utf-8")
            logger.info("Experiment report generated at %s", report_path)
        except Exception as exc:
            logger.error("Failed to generate experiment report at %s: %s", report_path, exc)

        return report_path
