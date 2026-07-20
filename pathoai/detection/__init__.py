"""pathoai.detection — Object Detection Engine for Cell Detection (Milestone 7)."""

from pathoai.detection.coordinate_transform import CoordinateTransformer
from pathoai.detection.evaluator import DetectionEvaluator
from pathoai.detection.exporter import (
    export_to_coco,
    export_to_csv,
    export_to_json,
    export_to_yolo_txt,
)
from pathoai.detection.factory import create_detector
from pathoai.detection.inference import DetectionInference
from pathoai.detection.merger import DetectionMerger
from pathoai.detection.metrics import DetectionMetrics
from pathoai.detection.model import DetectionModel
from pathoai.detection.pipeline import DetectionPipeline
from pathoai.detection.postprocessing import apply_nms, compute_iou
from pathoai.detection.registry import (
    get_detector_class,
    list_registered_detectors,
    register_detector,
)
from pathoai.detection.summary import generate_detector_summary
from pathoai.detection.tiling import TileGenerator, TileMetadata
from pathoai.detection.trainer import DetectionTrainer
from pathoai.detection.visualization import (
    create_density_heatmap,
    draw_detection_overlay,
)

__all__ = [
    "register_detector",
    "get_detector_class",
    "list_registered_detectors",
    "create_detector",
    "DetectionModel",
    "TileGenerator",
    "TileMetadata",
    "DetectionInference",
    "compute_iou",
    "apply_nms",
    "DetectionMerger",
    "CoordinateTransformer",
    "export_to_json",
    "export_to_csv",
    "export_to_coco",
    "export_to_yolo_txt",
    "draw_detection_overlay",
    "create_density_heatmap",
    "DetectionMetrics",
    "DetectionEvaluator",
    "DetectionTrainer",
    "generate_detector_summary",
    "DetectionPipeline",
]
