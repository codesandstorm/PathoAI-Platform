# Clinical Digital Pathology Platform UI Architecture (Milestone 11)

The Clinical Digital Pathology Platform provides an enterprise clinical application shell designed for hospital pathology departments, cancer centers, and computational pathology researchers.

---

## 🏗️ Architecture & Philosophy

- **Aesthetic Guidelines**: Minimal, clean, spacious, high information density, calm modern SaaS ("Linear + Vercel + Figma + Stripe").
- **Core Technology Stack**: Next.js / React SPA architecture, Tailwind CSS v4, Inter typography scale, 8px grid spacing system, and FastAPI REST API backend integration.

---

## 📱 Navigation Shell & Views

1. **Command Center (`/dashboard`)**: Enterprise operational metrics, GPU cluster load, slide statistics, active processing jobs, and recent clinical cases.
2. **Case Management (`/cases`)**: Searchable, filterable patient case table with sTIL percentages, 95% confidence intervals, diagnosis tags, and assigned pathologists.
3. **Interactive Whole Slide Viewer (`/viewer`)**: Deep-zoom canvas viewport with pan, zoom, scale bar ($100\,\mu\text{m}$), minimap, magnification indicators ($40\times, 20\times, 10\times$), multi-layer AI overlays (Tumor Bed, Stroma, Lymphocytes, ROIs, Heatmap, Confidence), opacity controls, right clinical metadata panel, and bottom stage timeline logs.
4. **AI Pipeline Inspector (`/analysis`)**: Expandable multi-stage pipeline flow (WSI $\to$ Segmentation $\to$ Tumor Bulk $\to$ Cell Detection $\to$ Spatial Fusion $\to$ sTIL Scoring $\to$ Validation), with per-stage execution times, memory/GPU usage, and intermediate outputs.
5. **Scientific Validation Dashboard (`/validation`)**: Stage-wise metrics (Dice, IoU, AP50, mAP50-95, ICC, Bland-Altman bias/limits, Pearson $r$, Spearman $\rho$, MAE, RMSE) with interactive scatter plot views, Bland-Altman agreement plots, PR curves, and error analysis outlier tables.
6. **Experiment Center (`/experiments`)**: MLflow / Weights & Biases style experiment tracking leaderboard comparing runs, git commits, seeds, backbones, and metrics across runs.
7. **Publication & Report Center (`/publication` & `/reports`)**: One-click generation of LaTeX tables, Markdown reports, publication figures, and downloadable PDF-ready clinical reports.
