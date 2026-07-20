# Clinical Rules Engine

The Clinical Rules Engine ([pathoai/scoring/clinical_rules.py](file:///d:/Research/PathoAI-Platform/pathoai/scoring/clinical_rules.py)) implements guideline-based threshold rules for sTIL score risk stratification.

---

## 🎯 Default Categorization Thresholds

| Clinical Category | sTIL Score Range (%) | Clinical Interpretation |
| :--- | :--- | :--- |
| **Low** | $< 10.0\%$ | Immune-cold tumor stroma. |
| **Intermediate** | $10.0\% - 49.9\%$ | Moderate stromal immune infiltration. |
| **High** | $\ge 50.0\%$ | Immune-rich tumor stroma; favorable response predictor. |

Thresholds are fully configurable via `ClinicalRules(low_threshold, high_threshold)`.
