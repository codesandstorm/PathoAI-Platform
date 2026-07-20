# Coordinate Transformations

The `CoordinateTransformer` ([pathoai/detection/coordinate_transform.py](file:///d:/Research/PathoAI-Platform/pathoai/detection/coordinate_transform.py)) converts detection coordinates across spatial reference frames:

$$\text{Tile Pixels} \longrightarrow \text{ROI Pixels} \longrightarrow \text{Slide Level-0 Pixels} \longrightarrow \text{Physical Microns }(\mu m)$$

---

## 📐 Transformation Equations

1. **Tile to Slide Conversion**:
   $$x_{slide} = x_{tile} + x_{tile\_offset}$$
   $$y_{slide} = y_{tile} + y_{tile\_offset}$$

2. **Centroid Calculation**:
   $$c_x = \frac{x_1 + x_2}{2.0}, \quad c_y = \frac{y_1 + y_2}{2.0}$$

3. **Physical Area Calculation ($\mu m^2$)**:
   $$\text{Area}_{\mu m^2} = (w_{px} \times \text{MPP}) \times (h_{px} \times \text{MPP})$$
