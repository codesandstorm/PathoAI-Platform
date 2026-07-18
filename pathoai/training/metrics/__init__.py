"""pathoai.training.metrics — performance metric engines.

Exposes:
    SegmentationMetrics: Calculates pixel accuracy, Dice, IoU, recall, precision, and F1.
    ConfusionMatrixMetric: Calculates raw and normalized confusion matrix and Cohen's Kappa.
    MetricCollection: Aggregates multiple metrics calculators.
"""

from pathoai.training.metrics.aggregation import MetricCollection
from pathoai.training.metrics.confusion import ConfusionMatrixMetric
from pathoai.training.metrics.segmentation import SegmentationMetrics

__all__ = [
    "SegmentationMetrics",
    "ConfusionMatrixMetric",
    "MetricCollection",
]
