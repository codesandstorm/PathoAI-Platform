"""pathoai.datasets — Dataset Engine for semantic segmentation and detection models.

Exposes:
    generate_dataset_manifest: Discovers WSIs and generates a patch manifest.
    parse_patient_id: Extracts patient IDs from slide names.
    apply_patient_split: Splits manifest entries patient-wise.
    compute_class_frequencies: Accumulates pixel counts per class.
    calculate_class_loss_weights: Calculates weights for class-imbalanced losses.
    generate_dataset_statistics_report: Formulates statistical split profiles.
    get_transforms: Returns Albumentations image augmentations.
    SegmentationDataset: PyTorch Dataset for semantic patch loading.
    WSICache: LRU slide reader cache.
    get_segmentation_dataloaders: PyTorch DataLoader factories.
"""

from pathoai.datasets.dataset import SegmentationDataset, WSICache
from pathoai.datasets.loader import get_segmentation_dataloaders
from pathoai.datasets.manifest import generate_dataset_manifest
from pathoai.datasets.split import apply_patient_split, parse_patient_id
from pathoai.datasets.statistics import (
    calculate_class_loss_weights,
    compute_class_frequencies,
    generate_dataset_statistics_report,
)
from pathoai.datasets.transforms import get_transforms

__all__ = [
    "SegmentationDataset",
    "WSICache",
    "get_segmentation_dataloaders",
    "generate_dataset_manifest",
    "apply_patient_split",
    "parse_patient_id",
    "calculate_class_loss_weights",
    "compute_class_frequencies",
    "generate_dataset_statistics_report",
    "get_transforms",
]
"""
pathoai/datasets/__init__.py
============================
Dataset package initializer.
"""
