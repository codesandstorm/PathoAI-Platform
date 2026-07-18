"""
pathoai/datasets/split.py
=========================
Patient-wise data partitioner to prevent data leakage.

Extracts patient IDs from slide names (TCGA & custom schemas) and partitions
patients (and all their slide patches) into disjoint train/val/test splits.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)


def parse_patient_id(filename: str) -> str:
    """Extract the patient ID from a slide filename.

    Supports TCGA nomenclature (first 12 characters: e.g. 'TCGA-OL-A5RW')
    and falls back to parsing standard dash/dot delimiters.

    Parameters
    ----------
    filename : str
        Slide filename (e.g. 'TCGA-OL-A5RW-01Z-00-DX1.tif').

    Returns
    -------
    str
        Parsed patient ID.
    """
    stem = Path(filename).name

    # Check for TCGA naming pattern: e.g., TCGA-XX-XXXX-XXX...
    # The first 12 characters contain "TCGA-XX-XXXX" (patient identifier)
    if stem.upper().startswith("TCGA-") and len(stem) >= 12:
        return stem[:12]

    # Fallback: split on first dash or dot
    parts = re_split_delimiters(stem)
    if parts:
        return parts[0]

    return stem


def re_split_delimiters(text: str) -> List[str]:
    """Helper to split string by dashes or dots."""
    import re
    return re.split(r"[-.]", text)


def apply_patient_split(
    manifest_entries: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assign patient-wise split labels to manifest entries and return split groups.

    Guarantees that all patches from the same patient fall into the same split,
    preventing slide-level or patch-level data leakage.

    Parameters
    ----------
    manifest_entries : List[Dict[str, Any]]
        List of manifest entries.
    train_ratio : float
        Fraction of patients in training.
    val_ratio : float
        Fraction of patients in validation.
    test_ratio : float
        Fraction of patients in test.
    seed : int
        Seeded random generator for reproducibility.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]
        Grouped manifest lists: (train_entries, val_entries, test_entries).

    Raises
    ------
    ValidationError
        If ratios don't sum to 1.0 or if patient assignment leakage is detected.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-4:
        raise ValidationError(
            f"Split ratios must sum to 1.0. Got: {train_ratio} + {val_ratio} + {test_ratio} "
            f"= {train_ratio + val_ratio + test_ratio}"
        )

    # 1. Map slides to patient IDs and inject patient_id into entries
    patient_to_entries: Dict[str, List[Dict[str, Any]]] = {}
    for entry in manifest_entries:
        slide_path = Path(entry["slide_path"])
        patient_id = parse_patient_id(slide_path.name)
        entry["patient_id"] = patient_id

        if patient_id not in patient_to_entries:
            patient_to_entries[patient_id] = []
        patient_to_entries[patient_id].append(entry)

    patients = sorted(list(patient_to_entries.keys()))
    n_patients = len(patients)

    if n_patients == 0:
        return [], [], []

    # 2. Shuffle patients deterministically
    rng = random.Random(seed)
    shuffled_patients = list(patients)
    rng.shuffle(shuffled_patients)

    # 3. Calculate partition sizes
    n_train = int(round(train_ratio * n_patients))
    n_val = int(round(val_ratio * n_patients))

    # Adjust partition bounds to be at least 1 if possible
    if n_patients >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)

    # Ensure test captures remainder
    n_test = n_patients - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n_patients - n_train

    # Slice patient sets
    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train:n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val:])

    # 4. Assert zero overlap (leakage check)
    overlap_tr_val = train_patients & val_patients
    overlap_tr_te = train_patients & test_patients
    overlap_val_te = val_patients & test_patients

    if overlap_tr_val or overlap_tr_te or overlap_val_te:
        raise ValidationError(
            f"Leakage detected in patient split sets: "
            f"tr_val: {overlap_tr_val}, tr_te: {overlap_tr_te}, val_te: {overlap_val_te}"
        )

    train_entries: List[Dict[str, Any]] = []
    val_entries: List[Dict[str, Any]] = []
    test_entries: List[Dict[str, Any]] = []

    # 5. Populate and label splits
    for patient_id, entries in patient_to_entries.items():
        if patient_id in train_patients:
            split_label = "train"
            split_list = train_entries
        elif patient_id in val_patients:
            split_label = "val"
            split_list = val_entries
        else:
            split_label = "test"
            split_list = test_entries

        for entry in entries:
            entry["split"] = split_label
            split_list.append(entry)

    logger.info(
        "Patient-wise split completed",
        extra={
            "n_patients": n_patients,
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "test_patients": len(test_patients),
            "train_patches": len(train_entries),
            "val_patches": len(val_entries),
            "test_patches": len(test_entries),
        },
    )

    return train_entries, val_entries, test_entries
