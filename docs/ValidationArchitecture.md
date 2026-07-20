# Validation & Benchmarking Architecture (Milestone 10)

The Scientific Validation & Benchmarking Framework (`pathoai/validation/`) provides multi-stage, statistically rigorous evaluation across all platform stages (Segmentation, Cell Detection, Spatial Fusion, and Clinical sTIL Scoring).

---

## 🏗️ Core Architecture Components

```
Ground Truth + Pipeline Predictions
               │
               ▼
       ValidationPipeline
               ├── Stage Evaluators (Segmentation, Detection, Fusion, Scoring)
               ├── CorrelationEngine (Pearson r, Spearman rho, R²)
               ├── AgreementEngine (ICC, Bland–Altman limits)
               ├── CalibrationEngine (ECE & Reliability diagrams)
               ├── BenchmarkEngine (Comparison vs literature baselines)
               ├── ErrorAnalysisEngine (Outliers & failure categorization)
               ├── ValidationVisualizer (Bland–Altman, Scatter, PR curves)
               └── ValidationExporter (JSON, CSV, Markdown reports)
```

---

## 🔑 Key Features

1. **Zero Model Inference Overhead**: Pure evaluation & statistical validation framework.
2. **Clinical Agreement Metrics**: Calculates Intraclass Correlation Coefficient (ICC) and Bland–Altman 95% limits of agreement.
3. **Multi-Stage Evaluation**: Evaluates Dice, IoU, mAP@50-95, MAE, RMSE, Pearson $r$, Spearman $\rho$, $R^2$, and ICC.
4. **Publication-Ready Figures & Reports**: Renders Bland–Altman plots, scatter plots, PR curves, and exports structured reports.
