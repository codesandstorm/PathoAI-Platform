# Detection Metrics

The `DetectionEvaluator` ([pathoai/validation/detection.py](file:///d:/Research/PathoAI-Platform/pathoai/validation/detection.py)) measures single-cell localization and classification performance.

---

## 🎯 Evaluated Metrics

- **Precision, Recall, F1 Score**.
- **AP@50**: Average Precision at IoU threshold 0.50.
- **AP@75**: Average Precision at IoU threshold 0.75.
- **mAP@50-95**: Mean Average Precision averaged over IoU thresholds 0.50 to 0.95.
- **Confusion Counts**: True Positives (TP), False Positives (FP), False Negatives (FN).
