# Statistics Engine

The Statistics Engine ([pathoai/scoring/statistics.py](file:///d:/Research/PathoAI-Platform/pathoai/scoring/statistics.py)) computes numerical cell counts, physical stroma areas in $mm^2$, and lymphocyte density ($cells/mm^2$) directly from `FusionResult` objects.

---

## 📊 Equations & Physical Units

- **Physical Stroma Area**:
  $$\text{Area}_{mm^2} = \frac{\text{Area}_{\mu m^2}}{1,000,000}$$
- **Lymphocyte Density**:
  $$\text{Density} = \frac{\text{Stromal Lymphocyte Count}}{\text{Area}_{mm^2}}\quad (\text{cells}/mm^2)$$
- **sTIL Percentage**:
  $$\text{sTIL \%} = \min\left(100.0, \frac{\text{Stromal Lymphocyte Count} \times (\pi \cdot r^2)}{\text{Area}_{\mu m^2}} \times 100\right)$$
