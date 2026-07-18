"""
pathoai/training/exporters/metrics_exporter.py
=============================================
Metrics Exporter.

Writes final evaluation metrics to JSON and CSV formats.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.8
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class MetricsExporter:
    """Exports evaluation metrics to JSON and CSV formats."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def export(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """Export metrics to files.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary of metrics keys and values.
        prefix : str
            File name prefix (e.g. 'val_' or 'test_').
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / f"{prefix}metrics.json"
        csv_path = self.output_dir / f"{prefix}metrics.csv"
        class_csv_path = self.output_dir / f"{prefix}class_metrics.csv"

        # 1. Save JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.debug("Metrics exported to JSON at %s", json_path)
        except Exception as exc:
            logger.error("Failed to export metrics to JSON: %s", exc)

        # 2. Save CSV (flattened scalars)
        try:
            scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}
            df = pd.DataFrame([scalars])
            df.to_csv(csv_path, index=False)
            logger.debug("Metrics exported to CSV at %s", csv_path)
        except Exception as exc:
            logger.error("Failed to export metrics to CSV: %s", exc)

        # 3. Save Class metrics (for Dice and IoU per class list)
        try:
            dice_list = metrics.get("dice_per_class")
            iou_list = metrics.get("iou_per_class")
            support_list = metrics.get("support_per_class")

            if dice_list is not None and iou_list is not None:
                class_rows = []
                for c in range(len(dice_list)):
                    row = {
                        "class_id": c,
                        "dice": dice_list[c],
                        "iou": iou_list[c],
                        "support_pixels": support_list[c] if support_list is not None else 0,
                    }
                    class_rows.append(row)

                class_df = pd.DataFrame(class_rows)
                class_df.to_csv(class_csv_path, index=False)
                logger.debug("Class-specific metrics exported to CSV at %s", class_csv_path)
        except Exception as exc:
            logger.error("Failed to export class-specific metrics to CSV: %s", exc)
