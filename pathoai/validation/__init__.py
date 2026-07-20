"""pathoai.validation — Scientific Validation & Benchmarking Framework (Milestone 10)."""

from pathoai.core.types import (
    BenchmarkResults,
    DetectionMetrics,
    ErrorAnalysis,
    ScoringMetrics,
    SegmentationMetrics,
    StatisticalAnalysis,
    ValidationReport,
    ValidationResult,
)
from pathoai.validation.ablation import AblationEngine
from pathoai.validation.agreement import AgreementEngine
from pathoai.validation.benchmark import BenchmarkEngine
from pathoai.validation.calibration import CalibrationEngine
from pathoai.validation.correlation import CorrelationEngine
from pathoai.validation.dataset_audit import DatasetAuditReport, audit_dataset
from pathoai.validation.dataset_validator import DatasetValidationReport, validate_dataset
from pathoai.validation.detection import DetectionEvaluator
from pathoai.validation.error_analysis import ErrorAnalysisEngine
from pathoai.validation.exporter import (
    export_validation_report_to_markdown,
    export_validation_result_to_json,
)
from pathoai.validation.factory import create_evaluator
from pathoai.validation.fusion import SpatialFusionEvaluator
from pathoai.validation.pipeline import ValidationPipeline
from pathoai.validation.registry import (
    get_evaluator_class,
    list_registered_evaluators,
    register_evaluator,
)
from pathoai.validation.report import ValidationReportGenerator
from pathoai.validation.robustness import RobustnessEngine
from pathoai.validation.scoring import ClinicalScoringEvaluator
from pathoai.validation.segmentation import SegmentationEvaluator
from pathoai.validation.statistics import ValidationStatistics
from pathoai.validation.summary import generate_validation_summary
from pathoai.validation.visualization import ValidationVisualizer

__all__ = [
    "validate_dataset",
    "audit_dataset",
    "DatasetValidationReport",
    "DatasetAuditReport",
    "SegmentationMetrics",
    "DetectionMetrics",
    "ScoringMetrics",
    "StatisticalAnalysis",
    "BenchmarkResults",
    "ErrorAnalysis",
    "ValidationResult",
    "ValidationReport",
    "register_evaluator",
    "get_evaluator_class",
    "list_registered_evaluators",
    "create_evaluator",
    "SegmentationEvaluator",
    "DetectionEvaluator",
    "SpatialFusionEvaluator",
    "ClinicalScoringEvaluator",
    "CorrelationEngine",
    "AgreementEngine",
    "ValidationStatistics",
    "CalibrationEngine",
    "RobustnessEngine",
    "BenchmarkEngine",
    "AblationEngine",
    "ErrorAnalysisEngine",
    "ValidationVisualizer",
    "ValidationReportGenerator",
    "ValidationPipeline",
    "export_validation_result_to_json",
    "export_validation_report_to_markdown",
    "generate_validation_summary",
]
