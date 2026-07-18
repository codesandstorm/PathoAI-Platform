"""
tests/unit/datasets/test_split.py
=================================
Unit tests for the patient splitting logic.

Tests cover:
- parse_patient_id with TCGA format and custom delimiter fallbacks
- apply_patient_split patient grouping and ratio partitioning
- Overlap/leakage detection (Train & Val & Test intersections must be empty)
- Ratio validation error checking

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

import pytest

from pathoai.core.exceptions import ValidationError
from pathoai.datasets.split import apply_patient_split, parse_patient_id


class TestPatientParsing:
    """Verifies that patient IDs are parsed correctly from filenames."""

    def test_tcga_nomenclature(self):
        filename = "TCGA-OL-A5RW-01Z-00-DX1.11111111-2222-3333-4444-555555555555.tif"
        assert parse_patient_id(filename) == "TCGA-OL-A5RW"

    def test_custom_naming_with_dashes(self):
        filename = "patient001-slideA.tif"
        assert parse_patient_id(filename) == "patient001"

    def test_custom_naming_with_dots(self):
        filename = "patient99.slideB.svs"
        assert parse_patient_id(filename) == "patient99"

    def test_single_word_name(self):
        filename = "normal.tif"
        assert parse_patient_id(filename) == "normal"


class TestPatientSplit:
    """Verifies patient-wise partitioning and data leakage checks."""

    def _make_dummy_manifest(self, num_patients: int, patches_per_patient: int = 3) -> list[dict]:
        manifest = []
        for p in range(num_patients):
            patient_id = f"TCGA-00-{p:04d}"
            for patch in range(patches_per_patient):
                manifest.append({
                    "slide_path": f"/data/images/{patient_id}-DX1.tif",
                    "x_level0": patch * 512,
                    "y_level0": 0,
                    "patch_size": 512,
                    "target_mpp": 0.50,
                })
        return manifest

    def test_apply_split_distributes_correctly(self):
        # 10 patients, 3 patches each = 30 patches total
        manifest = self._make_dummy_manifest(num_patients=10, patches_per_patient=3)
        train, val, test = apply_patient_split(
            manifest,
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            seed=42,
        )

        # Totals match
        assert len(train) + len(val) + len(test) == len(manifest)

        # Collect unique patients per split
        train_p = {e["patient_id"] for e in train}
        val_p = {e["patient_id"] for e in val}
        test_p = {e["patient_id"] for e in test}

        # Verify zero patient overlap (no leakage)
        assert train_p.isdisjoint(val_p)
        assert train_p.isdisjoint(test_p)
        assert val_p.isdisjoint(test_p)

        # Check ratio counts (10 patients: 6 train, 2 val, 2 test)
        assert len(train_p) == 6
        assert len(val_p) == 2
        assert len(test_p) == 2

        # Check that splits labels are correctly set in entries
        assert all(e["split"] == "train" for e in train)
        assert all(e["split"] == "val" for e in val)
        assert all(e["split"] == "test" for e in test)

    def test_ratios_not_summing_to_one_raises(self):
        manifest = self._make_dummy_manifest(5)
        with pytest.raises(ValidationError, match="Split ratios must sum to 1.0"):
            apply_patient_split(manifest, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)

    def test_handles_fewer_patients_gracefully(self):
        # 2 patients
        manifest = self._make_dummy_manifest(num_patients=2)
        train, val, test = apply_patient_split(
            manifest,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        assert len(train) > 0
        # If patients are very few, partitions adjust to fit
        assert len(train) + len(val) + len(test) == len(manifest)
