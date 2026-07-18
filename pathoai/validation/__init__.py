"""pathoai.validation — Dataset integrity and statistical validation engine.

Exposes:
    validate_dataset: Run full structural dataset validation.
    audit_dataset: Run statistical dataset profile audit.
    DatasetValidationReport: Report type from structural validation.
    DatasetAuditReport: Report type from statistical audit.
"""

from pathoai.validation.dataset_audit import DatasetAuditReport, audit_dataset
from pathoai.validation.dataset_validator import DatasetValidationReport, validate_dataset

__all__ = [
    "validate_dataset",
    "audit_dataset",
    "DatasetValidationReport",
    "DatasetAuditReport",
]
