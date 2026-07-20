# Interactive Whole Slide Image Viewer

The WSI Viewer ([pathoai/dashboard/ui/app.js](file:///d:/Research/PathoAI-Platform/pathoai/dashboard/ui/app.js)) mimics clinical workstations (Aperio, PathPresenter, QuPath).

---

## 🔍 Features

1. **Deep Zoom Viewport**: Supports zoom level toggles ($40\times, 20\times, 10\times$), pan controls, scale bar ($100\,\mu\text{m}$), and minimap.
2. **Multi-Layer AI Overlays**:
   - Tumor Bed Mask (Toggle ON/OFF, Opacity Slider)
   - Stroma Region Mask (Toggle ON/OFF, Opacity Slider)
   - Lymphocyte Detections (Toggle ON/OFF, Opacity Slider)
   - Tumor ROIs & Boundaries (Toggle ON/OFF)
   - sTIL Density Heatmap (Toggle ON/OFF)
3. **Right Metadata Panel**: Displays Stromal Area ($\text{mm}^2$), Stromal Lymphocyte Count, Lymphocyte Density ($\text{cells}/\text{mm}^2$), sTIL Percentage, 95% Confidence Interval, and Clinical Risk Classification.
4. **Bottom Pipeline Stage Timeline Log**: Displays step-by-step pipeline execution timestamps.
