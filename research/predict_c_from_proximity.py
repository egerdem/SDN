"""
Predict Optimal C from Wall Proximity (H_src_norm)

This script analyzes the Monte Carlo results to establish a predictive relationship
between normalized source wall proximity and the optimal C parameter for SDN.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)


# =====================================================================
# 1. LOAD DATA
# =====================================================================

RESULTS_FILE = os.path.join(project_root, "results", "monte_carlo_proximity_results.json")

with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

print(f"Loaded {len(results)} Monte Carlo runs")

# =====================================================================
# 2. IDENTIFY OUTLIERS
# =====================================================================

# Extract data
h_src_norms = np.array([r['h_src_norm'] for r in results])
opt_cs = np.array([r['optimal_c'] for r in results])
rmses = np.array([r['min_rmse'] for r in results])

# Define outliers as cases where:
# 1. C is at or near upper bound (≥ 6.9)
# 2. RMSE is high (> 0.4)

C_UPPER_THRESHOLD = 6.9
RMSE_THRESHOLD = 0.4

outlier_mask = (opt_cs >= C_UPPER_THRESHOLD) & (rmses > RMSE_THRESHOLD)
inlier_mask = ~outlier_mask

print(f"\n--- OUTLIER ANALYSIS ---")
print(f"Total runs: {len(results)}")
print(f"Outliers (C≥{C_UPPER_THRESHOLD} & RMSE>{RMSE_THRESHOLD}): {np.sum(outlier_mask)}")
print(f"Inliers: {np.sum(inlier_mask)}")

print(f"\nOutlier Details:")
for i, r in enumerate(results):
    if outlier_mask[i]:
        print(f"  Run {r['run_id']}: H_src_norm={r['h_src_norm']:.3f}, "
              f"C_opt={r['optimal_c']:.2f}, RMSE={r['min_rmse']:.3f}")

print(f"\n--- OUTLIERs removed ---")
print("1. C_opt hit the upper bound (7.0), suggesting optimizer couldn't find optimum")
print("2. RMSE is significantly higher (>0.4) than typical good fits (<0.3)")


# =====================================================================
# 3. FIT PREDICTIVE MODELS
# =====================================================================

X_all = h_src_norms.reshape(-1, 1)
y_all = opt_cs

X_inlier = h_src_norms[inlier_mask].reshape(-1, 1)
y_inlier = opt_cs[inlier_mask]

print(f"\n--- MODEL FITTING (Inliers Only) ---")

# Model 1: Linear Regression
model_linear = LinearRegression()
model_linear.fit(X_inlier, y_inlier)
y_pred_linear = model_linear.predict(X_inlier)
r2_linear = r2_score(y_inlier, y_pred_linear)
mae_linear = mean_absolute_error(y_inlier, y_pred_linear)

print(f"\nLinear Model: C = {model_linear.intercept_:.4f} + {model_linear.coef_[0]:.4f} * H_src_norm")
print(f"  R² = {r2_linear:.4f}")
print(f"  MAE = {mae_linear:.4f}")

# Model 2: Polynomial (Degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_inlier_poly = poly_features.fit_transform(X_inlier)
model_poly = LinearRegression()
model_poly.fit(X_inlier_poly, y_inlier)
y_pred_poly = model_poly.predict(X_inlier_poly)
r2_poly = r2_score(y_inlier, y_pred_poly)
mae_poly = mean_absolute_error(y_inlier, y_pred_poly)

print(f"\nPolynomial Model (Degree 2): C = {model_poly.intercept_:.4f} + "
      f"{model_poly.coef_[0]:.4f} * H + {model_poly.coef_[1]:.4f} * H²")
print(f"  R² = {r2_poly:.4f}")
print(f"  MAE = {mae_poly:.4f}")

# Model 3: Exponential-like (log transform)
# Try: C = a * exp(b * H_src_norm) => log(C) = log(a) + b * H_src_norm
# Only if all C > 0 (which they are, bounded at 1.0)
# Actually, let's try a power law: C = a * H_src_norm^b
# log(C) = log(a) + b * log(H_src_norm)
log_y_inlier = np.log(y_inlier)
log_X_inlier = np.log(X_inlier)

model_power = LinearRegression()
model_power.fit(log_X_inlier, log_y_inlier)
log_y_pred = model_power.predict(log_X_inlier)
y_pred_power = np.exp(log_y_pred)
r2_power = r2_score(y_inlier, y_pred_power)
mae_power = mean_absolute_error(y_inlier, y_pred_power)

a_power = np.exp(model_power.intercept_)
b_power = model_power.coef_[0]

print(f"\nPower Law Model: C = {a_power:.4f} * H_src_norm^{b_power:.4f}")
print(f"  R² = {r2_power:.4f}")
print(f"  MAE = {mae_power:.4f}")

# Choose best model
models = [
    ('Linear', model_linear, r2_linear, mae_linear),
    ('Polynomial', model_poly, r2_poly, mae_poly),
    ('Power', model_power, r2_power, mae_power)
]
best_model_name, best_model, best_r2, best_mae = max(models, key=lambda x: x[2])

print(f"\n*** Best Model: {best_model_name} (R²={best_r2:.4f}) ***")

# =====================================================================
# 4. VISUALIZATION
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: All data with outliers highlighted
ax1 = axes[0, 0]
sc1 = ax1.scatter(h_src_norms[inlier_mask], opt_cs[inlier_mask], 
                  c=rmses[inlier_mask], cmap='viridis', alpha=0.6, s=60, label='Inliers')
ax1.scatter(h_src_norms[outlier_mask], opt_cs[outlier_mask], 
           c='red', marker='x', s=100, linewidths=3, label='Outliers (C≥6.9, RMSE>0.4)')
ax1.set_xlabel('Normalized H_src (H_src / V^(1/3)) [Lower=Corner]', fontsize=11)
ax1.set_ylabel('Optimal C', fontsize=11)
ax1.set_title('Source Proximity vs Optimal C (Outliers Marked)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(sc1, ax=ax1, label='Min RMSE')

# Plot 2: Inliers with fitted models
ax2 = axes[0, 1]
ax2.scatter(h_src_norms[inlier_mask], opt_cs[inlier_mask], 
           c=rmses[inlier_mask], cmap='viridis', alpha=0.6, s=60)
ax2.set_xlabel('Normalized H_src', fontsize=11)
ax2.set_ylabel('Optimal C', fontsize=11)
ax2.set_title('Fitted Models (Inliers Only)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot model predictions
h_range = np.linspace(h_src_norms[inlier_mask].min(), h_src_norms[inlier_mask].max(), 100).reshape(-1, 1)

# Linear
c_pred_linear = model_linear.predict(h_range)
ax2.plot(h_range, c_pred_linear, 'r-', linewidth=2, label=f'Linear (R²={r2_linear:.3f})')

# Polynomial
h_range_poly = poly_features.transform(h_range)
c_pred_poly = model_poly.predict(h_range_poly)
ax2.plot(h_range, c_pred_poly, 'g--', linewidth=2, label=f'Poly-2 (R²={r2_poly:.3f})')

# Power
c_pred_power = a_power * h_range.flatten()**b_power
ax2.plot(h_range, c_pred_power, 'b:', linewidth=2, label=f'Power (R²={r2_power:.3f})')

ax2.legend()

# Plot 3: Residuals
ax3 = axes[1, 0]
if best_model_name == 'Linear':
    y_pred_best = model_linear.predict(X_inlier)
elif best_model_name == 'Polynomial':
    y_pred_best = model_poly.predict(X_inlier_poly)
else:  # Power
    y_pred_best = a_power * X_inlier.flatten()**b_power

residuals = y_inlier - y_pred_best
ax3.scatter(h_src_norms[inlier_mask], residuals, alpha=0.6, c=rmses[inlier_mask], cmap='viridis')
ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Normalized H_src', fontsize=11)
ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
ax3.set_title(f'Residuals for {best_model_name} Model', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Predicted vs Actual
ax4 = axes[1, 1]
ax4.scatter(y_inlier, y_pred_best, alpha=0.6, c=rmses[inlier_mask], cmap='viridis', s=60)
min_val = min(y_inlier.min(), y_pred_best.min())
max_val = max(y_inlier.max(), y_pred_best.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual C_optimal', fontsize=11)
ax4.set_ylabel('Predicted C_optimal', fontsize=11)
ax4.set_title(f'{best_model_name} Model: Predicted vs Actual', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "c_prediction_analysis.png"), dpi=150)
print(f"\nPlots saved to results/c_prediction_analysis.png")

# =====================================================================
# 5. CREATE PREDICTION FUNCTION
# =====================================================================

print(f"\n" + "="*70)
print("PREDICTION FUNCTION")
print("="*70)

# Save model parameters to JSON
model_params = {
    'model_type': best_model_name,
    'r2_score': float(best_r2),
    'mae': float(best_mae),
    'outlier_criteria': {
        'c_threshold': C_UPPER_THRESHOLD,
        'rmse_threshold': RMSE_THRESHOLD
    }
}

if best_model_name == 'Linear':
    model_params['intercept'] = float(model_linear.intercept_)
    model_params['coefficient'] = float(model_linear.coef_[0])

elif best_model_name == 'Polynomial':
    model_params['intercept'] = float(model_poly.intercept_)
    model_params['coef_h'] = float(model_poly.coef_[0])
    model_params['coef_h2'] = float(model_poly.coef_[1])


else:  # Power
    model_params['a'] = float(a_power)
    model_params['b'] = float(b_power)

# Save parameters
params_file = os.path.join(project_root, "results", "c_prediction_model_params.json")
with open(params_file, 'w') as f:
    json.dump(model_params, f, indent=2)
print(f"\nModel parameters saved to: {params_file}")

# =====================================================================
# 6. VALIDATION EXAMPLES
# =====================================================================

print(f"\n" + "="*70)
print("VALIDATION EXAMPLES")
print("="*70)

# Test on a few inlier cases
test_indices = np.random.choice(np.where(inlier_mask)[0], size=min(5, np.sum(inlier_mask)), replace=False)

print(f"\n{'Run':<6} {'H_src_norm':<12} {'Actual C':<10} {'Predicted C':<13} {'Error':<8} {'RMSE':<8}")
print("-" * 70)

for idx in test_indices:
    r = results[idx]
    h_norm = r['h_src_norm']
    actual_c = r['optimal_c']
    
    if best_model_name == 'Linear':
        pred_c = model_params['intercept'] + model_params['coefficient'] * h_norm
    elif best_model_name == 'Polynomial':
        pred_c = model_params['intercept'] + model_params['coef_h'] * h_norm + model_params['coef_h2'] * h_norm**2
    else:  # Power
        pred_c = model_params['a'] * h_norm ** model_params['b']
    
    pred_c = np.clip(pred_c, 1.0, 7.0)
    error = abs(actual_c - pred_c)
    
    print(f"{r['run_id']:<6} {h_norm:<12.4f} {actual_c:<10.3f} {pred_c:<13.3f} {error:<8.3f} {r['min_rmse']:<8.4f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

