# Scoring Architecture (Milestone 9)

The Clinical sTIL Scoring Engine (`pathoai/scoring/`) converts `FusionResult` objects into clinically interpretable, statistically validated, and explainable stromal Tumor Infiltrating Lymphocyte (sTIL) scores following International TIL Working Group guidelines.

---

## 🏗️ Core Architecture Components

```
FusionResult (Milestone 8)
        │
        ▼
  ScoringPipeline
        ├── StatisticsEngine (Physical area mm² & count calculations)
        ├── sTILScorer (International Working Group equation)
        ├── BootstrapEngine (95% Confidence Intervals via resampling)
        ├── ConfidenceEstimator (Uncertainty quality flags)
        ├── ClinicalRules & STILCategorizer (Low/Intermediate/High risk buckets)
        ├── STILExplainability (Transparent rationale generator)
        ├── ScoreValidator (Boundary & zero-division checks)
        ├── ScoreExporter (JSON, CSV, Markdown report formats)
        └── ReportGenerator (Assembles ClinicalReport DTOs)
```

---

## 🔑 Key Features

1. **Typed Domain Models**: `STILScore` and `ClinicalReport` DTOs encapsulate primary scores, stroma area ($mm^2$), density ($cells/mm^2$), confidence intervals, and clinical interpretations.
2. **Zero AI Inference Guarantee**: Strictly operates on non-AI spatial data DTOs (`FusionResult`).
3. **Bootstrap Resampling**: 500-iteration bootstrap resampling for precise empirical 95% Confidence Intervals.
4. **Explainable Clinical Reports**: Transparent human-readable explanations detailing cell counts, stroma area, excluded cells, and recommendations.
