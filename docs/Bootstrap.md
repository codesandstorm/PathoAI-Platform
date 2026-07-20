# Bootstrap Confidence Engine

The Bootstrap Confidence Engine ([pathoai/scoring/bootstrap.py](file:///d:/Research/PathoAI-Platform/pathoai/scoring/bootstrap.py)) estimates empirical 95% Confidence Intervals (`ci_lower`, `ci_upper`) for sTIL scores.

---

## 🎲 Resampling Protocol

1. **Input**: `FusionResult` and primary sTIL percentage score.
2. **Resampling**: Performs $N$ iterations (default $N=500$) of random sampling with replacement over spatial cell detections.
3. **Distribution**: Evaluates score variance across bootstrap samples.
4. **Percentile Interval**: Extracts empirical 2.5th and 97.5th percentiles to define the 95% Confidence Interval bounds.
