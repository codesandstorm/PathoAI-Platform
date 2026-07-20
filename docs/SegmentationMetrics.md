# Segmentation Metrics

The `SegmentationEvaluator` ([pathoai/validation/segmentation.py](file:///d:/Research/PathoAI-Platform/pathoai/validation/segmentation.py)) measures tissue region delineation quality.

---

## 📐 Evaluated Metrics

- **Dice Similarity Coefficient (DSC)**:
  $$\text{Dice} = \frac{2 \cdot |Y_{\text{true}} \cap Y_{\text{pred}}|}{|Y_{\text{true}}| + |Y_{\text{pred}}|}$$
- **Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{|Y_{\text{true}} \cap Y_{\text{pred}}|}{|Y_{\text{true}} \cup Y_{\text{pred}}|}$$
- **Precision, Recall, Specificity, Pixel Accuracy, F1 Score**.
