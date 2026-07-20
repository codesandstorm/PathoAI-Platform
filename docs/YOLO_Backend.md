# YOLO Detector Backend

The `YOLODetector` class ([pathoai/detection/architectures/yolo.py](file:///d:/Research/PathoAI-Platform/pathoai/detection/architectures/yolo.py)) is the initial registered object detector backend in PathoAI.

---

## 🛠️ Design & Integration

- Registered via `@register_detector("yolo")`.
- Accepts image tensors of shape `(B, 3, H, W)` and predicts bounding boxes `[x1, y1, x2, y2]`, `scores`, and `labels`.
- Built with a feature pyramid stem and grid decoding head.
- Easily swappable via configuration (`detection.architecture: "yolo"`).
