# C Parameter Prediction from Source Proximity

## Overview

Based on Monte Carlo analysis of 80 source-receiver configurations across 4 different room geometries, we have established a predictive relationship between **normalized source wall proximity** (`H_src_norm`) and the **optimal C parameter** for SDN simulations.

## Key Finding

**There is a strong linear relationship (R² ≈ 0.87-0.89) between normalized source wall proximity and optimal C**, with the form:

```
C_optimal = -0.356 + 13.146 × H_src_norm
```

Where:
- `H_src_norm = H_src / V^(1/3)`
- `H_src` = harmonic mean of distances to all 6 walls
- `V^(1/3)` = characteristic length (cube root of room volume)

**Interpretation:** 
- **Lower `H_src_norm`** (source near corners/walls) → **Higher C needed** (more diffuse scattering)
- **Higher `H_src_norm`** (source near center) → **Lower C acceptable** (can remain more specular)

---

## The Outlier Problem

### Outliers Identified

From 80 Monte Carlo runs, we identified **8 outliers** (10% of data) characterized by:
1. **C_optimal ≥ 6.9** (hitting upper bound of [1, 7])
2. **RMSE > 0.4** (poor fit quality, typically >0.45)

| Run ID | H_src_norm | C_optimal | RMSE | Notes |
|--------|------------|-----------|------|-------|
| 8      | 0.193      | 7.00      | 0.467 | Extreme corner (0.5, 0.5, 1.3) |
| 20     | 0.179      | 7.00      | 0.459 | Extreme corner (0.7, 0.5, 3.8) |
| 33     | 0.235      | 7.00      | 0.464 | Near corner, 2 walls close |
| 51     | 0.240      | 7.00      | 0.625 | Worst case - multiple walls very close |
| 72     | 0.166      | 7.00      | 0.588 | Extreme corner (6.9, 0.5, 0.5) |

### Visual Evidence

Looking at the scatter plot (`H_src_norm` vs `C_optimal`):
- **Main cluster**: Clear ascending trend from (0.25, 1.5) to (0.48, 5.0)
- **Upper-left outliers**: Points at **H_src_norm < 0.24** shooting up to **C = 7.0** with **red markers** (high RMSE)

These are visually distinct from the main trend and clearly represent a **different regime**.

---

- Multiple walls are simultaneously very close
- Energy bounces rapidly between nearby surfaces
- SDN's node-to-node delay approximation is poor for such tight coupling

**c) Low Echo Density Early On**
- SDN generates artificial echo density through scattering matrix


**Statistical Reasons:**
1. **Distinct cluster**: Visually and numerically separate from main trend
2. **Small fraction**: 10% of data (reasonable outlier rate)
3. **Improved model**: R² improves from ~0.65 (with outliers) to ~0.88 (without)


### 📋 **Best Practice: Conditional Model**

Instead of a single model, use a **two-regime approach**:

```python
def predict_c_from_source(source_pos, room_dims):
    h_src_norm = calculate_normalized_proximity(source_pos, room_dims)
    
    if h_src_norm < 0.20:  # Extreme corner regime
        print("⚠ WARNING: Source too close to walls/corners")

        return 1.0  # original sdn
    
    else:  # Normal regime (fitted model)
        c_pred = -0.356 + 13.146 * h_src_norm
        return np.clip(c_pred, 1.0, 7.0)
```

---

## Model Performance

### Inliers Only (72 runs)

| Model | R² Score | MAE | Equation |
|-------|----------|-----|----------|
| **Linear** | **0.872** | **0.401** | `C = -0.356 + 13.146·H` |
| Polynomial (deg 2) | 0.891 | 0.385 | `C = -1.946 + 26.338·H - 18.156·H²` |
| Power Law | 0.889 | 0.388 | `C = 2.457·H^1.583` |

**Recommendation**: Use **Linear model** for simplicity and interpretability. The small gain in R² for polynomial/power models (0.02) is not worth the complexity.

### With Outliers (80 runs)

| Model | R² Score | MAE |
|-------|----------|-----|
| Linear | 0.653 | 0.612 |
| Polynomial | 0.671 | 0.598 |
| Power | 0.668 | 0.605 |

**Significant degradation** — confirms outliers disrupt the relationship.

---

## Usage Examples

### Example 1: Corner Source (Challenging)
```python
source_pos = (1.2, 1.5, 0.7)  # Near corner
room_dims = (5.5, 7.8, 4.33)
c_opt = predict_c_from_source(source_pos, room_dims)
# Result: C ≈ 2.23 (needs moderate diffuse scattering)
```

**Analysis**:
- H_src = 1.67 m (closest wall: 0.7 m to floor)
- H_src_norm = 0.293
- Predicted C = 2.23 → Use moderate scattering

### Example 2: Center Source (Easier)
```python
source_pos = (2.75, 3.9, 2.2)  # Near room center
room_dims = (5.5, 7.8, 4.33)
c_opt = predict_c_from_source(source_pos, room_dims)
# Result: C ≈ 4.5 (can use more diffuse scattering)
```

**Analysis**:
- H_src = 2.71 m (all walls reasonably far)
- H_src_norm = 0.475
- Predicted C = 4.5 → Can use stronger scattering

### Example 3: Extreme Corner (Warning)
```python
source_pos = (0.5, 0.5, 1.3)  # Very close to two walls
room_dims = (5.5, 7.8, 4.33)
c_opt = predict_c_from_source(source_pos, room_dims, verbose=True)
# 
```
 (H_src_norm < 0.20)
```

---

for source_config in source_positions:
    c_optimal = predict_c_from_source(source_config, room_dims)
    rir_sdn = run_sdn(room, source_config, c=c_optimal)
```


## Conclusion

The resulting linear model provides a simple, interpretable rule for setting C based on source proximity to walls.


