# Clinical sTIL Validation & Agreement

The Clinical Scoring Evaluator ([pathoai/validation/scoring.py](file:///d:/Research/PathoAI-Platform/pathoai/validation/scoring.py)) evaluates AI-vs-Pathologist agreement.

---

## 🔬 Inter-Rater Agreement Metrics

1. **Intraclass Correlation Coefficient (ICC)**: Evaluates consistency across raters. Values $>0.85$ indicate excellent agreement.
2. **Bland–Altman Analysis**:
   - **Mean Bias**: Average difference $(\text{AI} - \text{Pathologist})$.
   - **95% Limits of Agreement**: $\text{Bias} \pm 1.96 \cdot \text{SD}_{\text{diff}}$.
3. **Pearson $r$ & Spearman $\rho$**: Evaluates linear and monotonic rank correlation.
4. **MAE & RMSE**: Evaluates absolute error bounds.
