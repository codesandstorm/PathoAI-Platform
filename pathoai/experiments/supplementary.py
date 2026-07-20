"""
pathoai/experiments/supplementary.py
=====================================
Supplementary Publication Package Generator.

Bundles supplementary figures, tables, confusion matrices, Bland–Altman plots, and manifests.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5.7
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from pathoai.core.types import ValidationReport, ValidationResult
from pathoai.experiments.latex import LaTeXExporter
from pathoai.experiments.manifest import ManifestGenerator
from pathoai.experiments.tables import PublicationTableGenerator


class SupplementaryPackageGenerator:
    """Generates complete supplementary research publication package."""

    def __init__(self) -> None:
        self.table_gen = PublicationTableGenerator()
        self.latex_exporter = LaTeXExporter()
        self.manifest_gen = ManifestGenerator()

    def generate_supplementary_package(
        self,
        report: ValidationReport,
        output_dir: Union[str, Path],
    ) -> Path:
        """Assembles supplementary folder with figures, tables, and manifest."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        res = report.validation_result

        # 1. Manifest
        manifest = self.manifest_gen.create_manifest(
            config=None or type("Config", (), {
                "experiment_id": report.experiment_name,
                "segmentation_model": "deeplabv3plus",
                "segmentation_version": "v1.0",
                "detection_model": "yolo",
                "detection_version": "v1.0",
                "scoring_version": "v1.0",
            })(),
            dataset_name=res.dataset_name,
        )
        self.manifest_gen.export_manifest_to_json(manifest, out / "manifest.json")

        # 2. Publication Tables
        t1 = self.table_gen.generate_table_1_segmentation(res)
        t2 = self.table_gen.generate_table_2_detection(res)
        t3 = self.table_gen.generate_table_3_agreement(res)

        with open(out / "publication_tables.md", "w", encoding="utf-8") as f:
            f.write(f"# Supplementary Tables\n\n{t1}\n\n{t2}\n\n{t3}\n")

        # 3. LaTeX tables
        latex_code = self.latex_exporter.export_latex_table_3_agreement(res)
        with open(out / "table3_agreement.tex", "w", encoding="utf-8") as f:
            f.write(latex_code)

        return out
