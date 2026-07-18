"""pathoai.training.visualization — training progress visualizers.

Exposes:
    plot_training_curves: Generates loss, learning rate, and metric curves.
    plot_confusion_matrix: Plots raw and normalized confusion matrix heatmaps.
    generate_prediction_gallery_row: Generates comparison gallery strips.
"""

from pathoai.training.visualization.confusion import plot_confusion_matrix
from pathoai.training.visualization.curves import plot_training_curves
from pathoai.training.visualization.overlays import generate_prediction_gallery_row

__all__ = [
    "plot_training_curves",
    "plot_confusion_matrix",
    "generate_prediction_gallery_row",
]
