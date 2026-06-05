"""
Predict Optimal C from Wall Proximity (H_src_norm)

This script analyzes Monte Carlo results to fit predictive models relating 
source wall proximity (h_src_norm) to the optimal C parameter for SDN.

What it does:
-------------
1. Loads results.json from a Monte Carlo experiment
2. Identifies and removes outliers using principled criteria:
   - C ≥ 6.9 (optimizer hit upper bound)
   - AND RMSE(C=1) < RMSE(C*) (baseline performs better)
   If rmse_c1 not in results.json,
3. Fits three models to h_src_norm vs optimal_c:
   - Linear: C = a + b*H
   - Polynomial: C = a + b*H + c*H²
   - Power: C = a * H^b
4. Generates plots comparing all three models
5. Saves fitted parameters to experiment folder

Note on Outlier Detection:
--------------------------
For principled outlier rejection, your results.json should include 'rmse_c1'
(RMSE at baseline C=1). If missing, the script will use a fallback threshold.
To add rmse_c1, modify monte_carlo_proximity.py to compute RMSE at C=1.

Outputs (saved to same folder as results.json):
-----------------------------------------------
- c_prediction_model_params.json (fitted model parameters)
- c_prediction_analysis.png (visualization of fits)

Usage:
------
1. Update RESULTS_FILE to point to your Monte Carlo experiment
2. Run: python research/predict_c_from_proximity.py
3. Check the output plots and fitted parameters
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
import matplotlib
matplotlib.use('Qt5Agg')  # Use non-interactive backend for saving plots
# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

# =====================================================================
# CONFIGURATION FLAGS
# =====================================================================
SAVE_JSON = True            # Save c_prediction_model_params.json to experiment folder
PLOT_FIRST_FIGURE = True    # Plot both: d_sm_norm and H_src_norm vs optimal_c
SAVE_FIGURE = True          # Save the plotted figure to experiment folder


# =====================================================================
# 1. LOAD DATA
# =====================================================================

# =====================================================================
# MONTE CARLO EXPERIMENT SELECTION
# =====================================================================
# Choose which Monte Carlo experiment to analyze:

# Option 1: Legacy 4x20 experiment (50% corner bias, no seed specified)
# RESULTS_FILE = os.path.join(project_root, "results", "monte_carlo_experiments",
#                             "legacy_4x20_50cornerplacement_noseed", "results.json")

#RESULTS_FILE = os.path.join(project_root, "results", "monte_carlo_experiments",
#                            "10x60_seed42_0cornerplacement_big_V2", "results.json")

RESULTS_FILE = os.path.join(project_root, "results", "paper_data",
                            "aes_fullgrid_7src_3Ddiagonal", "results.json")

# RESULTS_FILE = os.path.join(project_root, "results", "monte_carlo_experiments",
#                             "10x60_seed42_25cornerplacement_big", "results.json")

# Option 2: Default experiment (current, with seed=42)
# RESULTS_FILE = os.path.join(project_root, "results", "monte_carlo_experiments", 
#                             "default", "results.json")

# Option 3: Legacy flat file (old structure)
# RESULTS_FILE = os.path.join(project_root, "results", "monte_carlo_proximity_results.json")

# Option 4: AES fullgrid experiments
# RESULTS_FILE = os.path.join(project_root, "results", "paper_data/experiments",
#                             "aes_fullgrid_8src_diagonal", "results.json")


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
# 2. Baseline C=1 performs BETTER than the "optimal" C
#
# This is the principled outlier criterion: if optimizer hits bound
# but C=1 achieves lower RMSE, the optimization clearly failed.

C_UPPER_THRESHOLD = 6.9

# Check if results contain rmse_c1 (baseline RMSE)
has_rmse_c1 = 'rmse_c1' in results[0]

if has_rmse_c1:
    # Use principled comparison: C=1 performs better
    rmse_c1_vals = np.array([r['rmse_c1'] for r in results])
    outlier_mask = (opt_cs >= C_UPPER_THRESHOLD) & (rmse_c1_vals < rmses)
else:
    # Fallback:
    print("⚠️  WARNING: rmse_c1 not found in results.json")


inlier_mask = ~outlier_mask

print(f"\n--- OUTLIER ANALYSIS ---")
print(f"Total runs: {len(results)}")
if has_rmse_c1:
    print(f"Outliers (C≥{C_UPPER_THRESHOLD} & RMSE(C=1)<RMSE(C*)): {np.sum(outlier_mask)}")
    print(f"  Logic: Optimizer hit bound BUT baseline C=1 performs better")
else:
    print(f"Outliers (C≥{C_UPPER_THRESHOLD} only): {np.sum(outlier_mask)}")
    print(f"  Logic: Only C threshold (no baseline comparison available)")
print(f"Inliers: {np.sum(inlier_mask)}")

print(f"\nOutlier Details:")
for i, r in enumerate(results):
    if outlier_mask[i]:
        if has_rmse_c1:
            print(f"  Run {r['run_id']}: H_src_norm={r['h_src_norm']:.3f}, "
                  f"C_opt={r['optimal_c']:.2f}, RMSE(C*)={r['min_rmse']:.3f}, "
                  f"RMSE(C=1)={r['rmse_c1']:.3f}")
        else:
            print(f"  Run {r['run_id']}: H_src_norm={r['h_src_norm']:.3f}, "
                  f"C_opt={r['optimal_c']:.2f}, RMSE={r['min_rmse']:.3f}")

print(f"\n--- OUTLIER REMOVAL CRITERIA ---")
print("1. C_opt hit the upper bound (~7.0) → optimizer couldn't find minimum")
if has_rmse_c1:
    print("2. Baseline C=1 achieves LOWER RMSE than C* → optimization clearly failed")
else:
    print("2. RMSE is high (>0.4) → arbitrary threshold (less principled)")

# =====================================================================
# 2.5. APPLY OUTLIER CORRECTION TO JSON
# =====================================================================

print(f"\n--- APPLYING OUTLIER CORRECTION ---")
print("Replacing optimal_c with 1.0 for identified outliers...")

correction_count = 0
for i, r in enumerate(results):
    if outlier_mask[i]:
        # Store original value (if not already stored)
        if 'optimal_c_original' not in r:
            r['optimal_c_original'] = r['optimal_c']
            r['min_rmse_original'] = r['min_rmse']
        
        # Replace with C=1 values
        r['optimal_c'] = 1.0
        if has_rmse_c1:
            r['min_rmse'] = r['rmse_c1']  # Use baseline RMSE
        r['outlier_corrected'] = True
        correction_count += 1
        
        print(f"  Run {r['run_id']}: C={r['optimal_c_original']:.2f} → C=1.0, "
              f"RMSE={r['min_rmse_original']:.3f} → {r['min_rmse']:.3f}")

# Mark non-outliers explicitly
for i, r in enumerate(results):
    if not outlier_mask[i] and 'outlier_corrected' not in r:
        r['outlier_corrected'] = False

# Save modified JSON back to file
print(f"\nSaving corrected results back to: {RESULTS_FILE}")
import shutil
from datetime import datetime

# Create backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = RESULTS_FILE.replace(".json", f"_backup_{timestamp}.json")
shutil.copy2(RESULTS_FILE, backup_path)
print(f"  ✅ Created backup: {os.path.basename(backup_path)}")

# Save corrected data
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✅ Saved {len(results)} entries with {correction_count} outliers corrected")

# Update opt_cs array with corrected values
opt_cs = np.array([r['optimal_c'] for r in results])
rmses = np.array([r['min_rmse'] for r in results])

# Recalculate inlier mask AFTER correction - now includes corrected outliers at c=1
if has_rmse_c1:
    rmse_c1_vals = np.array([r['rmse_c1'] for r in results])
    outlier_mask_new = (opt_cs >= C_UPPER_THRESHOLD) & (rmse_c1_vals < rmses)
else:
    outlier_mask_new = (opt_cs >= C_UPPER_THRESHOLD)

inlier_mask = ~outlier_mask_new

print(f"\n--- MODEL FITTING ---")
print(f"Fitting on {np.sum(inlier_mask)} data points (includes {correction_count} corrected values at c=1)")


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

# Determine number of subplots based on PLOT_FIRST_FIGURE flag
if PLOT_FIRST_FIGURE:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1 = axes[0]
    ax2 = axes[1]
    
    # Plot 1: d_sm_norm vs optimal_c (NEW - shows source-mic distance relationship)
    # Extract d_sm_norm (only available for per_pair mode)
    has_d_sm = 'd_sm_norm' in results[0]
    if has_d_sm:
        d_sm_norms = np.array([r.get('d_sm_norm', np.nan) for r in results])
        
        # Get original C values for outliers (before correction)
        opt_cs_original = np.array([r.get('optimal_c_original', r['optimal_c']) for r in results])
        
        sc1 = ax1.scatter(d_sm_norms[inlier_mask], opt_cs[inlier_mask], 
                          c=rmses[inlier_mask], cmap='viridis', alpha=0.6, s=60, label='Inliers')
        if np.sum(outlier_mask) > 0:
            # Plot original outlier positions with red X
            ax1.scatter(d_sm_norms[outlier_mask], opt_cs_original[outlier_mask], 
                       c='red', marker='x', s=100, linewidths=3, label='Outliers (original)', zorder=5)
            # Plot corrected positions at c=1 with green circles
            ax1.scatter(d_sm_norms[outlier_mask], opt_cs[outlier_mask], 
                       c='lime', marker='o', s=80, edgecolors='darkgreen', linewidths=2, 
                       label='Corrected (c=1)', zorder=5, alpha=0.8)
        
        # Fit line to d_sm_norm vs c
        d_inlier = d_sm_norms[inlier_mask]
        c_inlier_d = opt_cs[inlier_mask]
        if len(d_inlier) > 1:
            coeffs_d = np.polyfit(d_inlier, c_inlier_d, 1)
            d_range = np.linspace(d_inlier.min(), d_inlier.max(), 100)
            c_fit_d = coeffs_d[1] + coeffs_d[0] * d_range
            ax1.plot(d_range, c_fit_d, 'r--', linewidth=2, alpha=0.7,
                    label=f'Fit: C = {coeffs_d[1]:.2f} + {coeffs_d[0]:.2f}·d_sm')
            
            # Compute correlation
            corr_d = np.corrcoef(d_inlier, c_inlier_d)[0, 1]
            ax1.text(0.05, 0.95, f'Correlation: {corr_d:+.3f}', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('Normalized Source-Mic Distance (d_sm / V^(1/3))', fontsize=11)
        ax1.set_ylabel('Optimal C', fontsize=11)
        ax1.set_title('Source-Mic Distance vs Optimal C', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(sc1, ax=ax1, label='Min RMSE')
    else:
        # Fallback if d_sm_norm not available: show H_src with outliers
        opt_cs_original = np.array([r.get('optimal_c_original', r['optimal_c']) for r in results])
        
        sc1 = ax1.scatter(h_src_norms[inlier_mask], opt_cs[inlier_mask], 
                          c=rmses[inlier_mask], cmap='viridis', alpha=0.6, s=60, label='Inliers')
        if np.sum(outlier_mask) > 0:
            # Plot original outlier positions
            ax1.scatter(h_src_norms[outlier_mask], opt_cs_original[outlier_mask], 
                       c='red', marker='x', s=100, linewidths=3, label='Outliers (original)', zorder=5)
            # Plot corrected positions at c=1
            ax1.scatter(h_src_norms[outlier_mask], opt_cs[outlier_mask], 
                       c='lime', marker='o', s=80, edgecolors='darkgreen', linewidths=2, 
                       label='Corrected (c=1)', zorder=5, alpha=0.8)
        ax1.set_xlabel('Normalized H_src (H_src / V^(1/3)) [Lower=Corner]', fontsize=11)
        ax1.set_ylabel('Optimal C', fontsize=11)
        ax1.set_title('Source Proximity vs Optimal C (Outliers Marked)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(sc1, ax=ax1, label='Min RMSE')
else:
    # Only plot the second figure (H_src_norm with fitted models)
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 7))

# Plot 2: H_src_norm with all three fitted models
ax2.scatter(h_src_norms[inlier_mask], opt_cs[inlier_mask], 
           c=rmses[inlier_mask], cmap='viridis', alpha=0.6, s=60)
ax2.set_xlabel('Normalized H_src', fontsize=11)
ax2.set_ylabel('Optimal c', fontsize=11)
# ax2.set_title('Fitted Models (Inliers Only)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot model predictions
h_range = np.linspace(h_src_norms[inlier_mask].min(), h_src_norms[inlier_mask].max(), 100).reshape(-1, 1)

# Linear
c_pred_linear = model_linear.predict(h_range)
ax2.plot(h_range, c_pred_linear, 'r-', linewidth=2.5, 
        label=f'Linear: c = {model_linear.intercept_:.2f} + {model_linear.coef_[0]:.2f}·H (R²={r2_linear:.3f})')

# Polynomial
h_range_poly = poly_features.transform(h_range)
c_pred_poly = model_poly.predict(h_range_poly)
# ax2.plot(h_range, c_pred_poly, 'g--', linewidth=2.5,
#         label=f'Poly: c = {model_poly.intercept_:.2f} + {model_poly.coef_[0]:.2f}·H + {model_poly.coef_[1]:.2f}·H² (R²={r2_poly:.3f})')

# Power
c_pred_power = a_power * h_range.flatten()**b_power
# ax2.plot(h_range, c_pred_power, 'b:', linewidth=2.5, 
#         label=f'Power: c = {a_power:.2f}·H^{b_power:.2f} (R²={r2_power:.3f})')

ax2.legend(fontsize=9, loc='best')

plt.tight_layout()

# Save to experiment directory (if SAVE_FIGURE is True)
if SAVE_FIGURE:
    output_dir = os.path.dirname(RESULTS_FILE)
    output_path = os.path.join(output_dir, "c_prediction_analysis.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nPlots saved to {output_path}")
else:
    print("\nFigure not saved (SAVE_FIGURE=False)")

# =====================================================================
# 5. CREATE PREDICTION FUNCTION
# =====================================================================

print(f"\n" + "="*70)
print("PREDICTION FUNCTION")
print("="*70)

# Save ALL three model parameters (not just best one)
all_models_params = {
    'outlier_criteria': {
        'c_threshold': C_UPPER_THRESHOLD,
        'logic': 'C≥6.9 & RMSE(C=1)<RMSE(C*)' if has_rmse_c1 else 'C≥6.9 only',
        'uses_baseline_comparison': has_rmse_c1,
        'n_outliers': int(np.sum(outlier_mask)),
        'n_inliers': int(np.sum(inlier_mask))
    },
    'models': {
        'linear': {
            'type': 'Linear',
            'equation': f'C = {model_linear.intercept_:.4f} + {model_linear.coef_[0]:.4f} * H_src_norm',
            'intercept': float(model_linear.intercept_),
            'coefficient': float(model_linear.coef_[0]),
            'r2_score': float(r2_linear),
            'mae': float(mae_linear)
        },
        'polynomial': {
            'type': 'Polynomial (degree 2)',
            'equation': f'C = {model_poly.intercept_:.4f} + {model_poly.coef_[0]:.4f}*H + {model_poly.coef_[1]:.4f}*H²',
            'intercept': float(model_poly.intercept_),
            'coef_h': float(model_poly.coef_[0]),
            'coef_h2': float(model_poly.coef_[1]),
            'r2_score': float(r2_poly),
            'mae': float(mae_poly)
        },
        'power': {
            'type': 'Power Law',
            'equation': f'C = {a_power:.4f} * H_src_norm^{b_power:.4f}',
            'a': float(a_power),
            'b': float(b_power),
            'r2_score': float(r2_power),
            'mae': float(mae_power)
        }
    },
    'best_model': best_model_name.lower()
}

# Save to experiment directory (same as RESULTS_FILE) if SAVE_JSON is True
if SAVE_JSON:
    output_dir = os.path.dirname(RESULTS_FILE)
    params_file = os.path.join(output_dir, "c_prediction_model_params.json")
    with open(params_file, 'w') as f:
        json.dump(all_models_params, f, indent=2)
    print(f"\nAll model parameters saved to: {params_file}")
    print(f"Best model: {best_model_name} (R²={best_r2:.4f})")
else:
    print(f"\nJSON not saved (SAVE_JSON=False)")
    print(f"Best model: {best_model_name} (R²={best_r2:.4f})")

# =====================================================================
# 6. VALIDATION EXAMPLES
# =====================================================================

print(f"\n" + "="*70)
print("VALIDATION EXAMPLES")
print("="*70)

# Test on a few inlier cases
test_indices = np.random.choice(np.where(inlier_mask)[0], size=min(5, np.sum(inlier_mask)), replace=False)

print(f"\n{'Run':<16} {'H_src_norm':<12} {'Actual C':<10} {'Predicted C':<13} {'Error':<8} {'RMSE':<8}")
print("-" * 70)

for idx in test_indices:
    r = results[idx]
    h_norm = r['h_src_norm']
    actual_c = r['optimal_c']
    
    # Use best model from all_models_params
    best_model_key = all_models_params['best_model']
    model_data = all_models_params['models'][best_model_key]
    
    if best_model_key == 'linear':
        pred_c = model_data['intercept'] + model_data['coefficient'] * h_norm
    elif best_model_key == 'polynomial':
        pred_c = model_data['intercept'] + model_data['coef_h'] * h_norm + model_data['coef_h2'] * h_norm**2
    else:  # power
        pred_c = model_data['a'] * h_norm ** model_data['b']
    
    pred_c = np.clip(pred_c, 1.0, 7.0)
    error = abs(actual_c - pred_c)
    
    run_identifier = str(r.get('run_id', r.get('run_key', 'N/A')))
    print(f"{run_identifier:<16} {h_norm:<12.4f} {actual_c:<10.3f} {pred_c:<13.3f} {error:<8.3f} {r['min_rmse']:<8.4f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

