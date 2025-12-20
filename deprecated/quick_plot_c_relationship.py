"""
Quick visualization of H_src_norm vs C_optimal relationship
Highlights outliers and shows fitted line
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# Add project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Load results

# results_file = os.path.join(project_root, "results", "monte_carlo_proximity_results.json")
results_file = os.path.join(project_root, "results", "paper_data/experiments/aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair/results.json")

with open(results_file, 'r') as f:
    results = json.load(f)

# Extract data
h_src_norms = np.array([r['h_src_norm'] for r in results])
opt_cs = np.array([r['optimal_c'] for r in results])
rmses = np.array([r['min_rmse'] for r in results])

# Define outliers
outlier_mask = (opt_cs >= 6.9) & (rmses > 0.4)
inlier_mask = ~outlier_mask

# Fit line to inliers
h_inlier = h_src_norms[inlier_mask]
c_inlier = opt_cs[inlier_mask]
coeffs = np.polyfit(h_inlier, c_inlier, 1)  # Linear fit

print("="*70)
print("C PARAMETER PREDICTION FROM SOURCE PROXIMITY")
print("="*70)
print(f"\nTotal runs: {len(results)}")
print(f"Inliers: {np.sum(inlier_mask)} (used for fit)")
print(f"Outliers: {np.sum(outlier_mask)} (excluded: C≥6.9 & RMSE>0.4)")
print(f"\nFitted Model (inliers only):")
print(f"  C_optimal = {coeffs[1]:.4f} + {coeffs[0]:.4f} × H_src_norm")
print(f"  R² = {np.corrcoef(h_inlier, c_inlier)[0,1]**2:.4f}")

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: All data with outliers marked
ax1.scatter(h_inlier, c_inlier, c=rmses[inlier_mask], cmap='viridis', 
            alpha=0.7, s=80, label='Inliers', edgecolors='black', linewidths=0.5)
ax1.scatter(h_src_norms[outlier_mask], opt_cs[outlier_mask], 
           c='red', marker='X', s=200, linewidths=2, 
           label=f'Outliers (n={np.sum(outlier_mask)})', edgecolors='darkred', zorder=10)

# Fitted line
h_range = np.linspace(h_inlier.min(), h_inlier.max(), 100)
c_fit = coeffs[1] + coeffs[0] * h_range
ax1.plot(h_range, c_fit, 'r--', linewidth=3, alpha=0.8,
         label=f'Fit: C = {coeffs[1]:.2f} + {coeffs[0]:.2f}·H')

ax1.set_xlabel('Normalized H_src (H_src / V^(1/3)) [Lower=Corner]', fontsize=13, fontweight='bold')
ax1.set_ylabel('Optimal C Parameter', fontsize=13, fontweight='bold')
ax1.set_title('Source Wall Proximity vs Optimal C\n(Monte Carlo Results)',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0.15, 0.50)
ax1.set_ylim(0.5, 7.5)

# Add annotations for outliers
for i, r in enumerate(results):
    if outlier_mask[i]:
        ax1.annotate(f"R{r['run_id']}", 
                    xy=(r['h_src_norm'], r['optimal_c']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='darkred', fontweight='bold')

# Add interpretation zones
# Use percentile-based zones (data-driven, not hard-coded)
# We compute terciles (33rd/66th percentiles) on the *inlier* C distribution.
q33, q66 = np.percentile(c_inlier, [33.33, 66.67])
ax1.axhspan(0.5, q33, alpha=0.1, color='green', zorder=0)
ax1.axhspan(q33, q66, alpha=0.1, color='yellow', zorder=0)
ax1.axhspan(q66, 7.5, alpha=0.1, color='red', zorder=0)

# Place labels at the midpoints of each band
x_label = 0.48
ax1.text(x_label, (0.5 + q33) / 2.0, f'Low C\n(≤ P33={q33:.2f})', ha='right', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax1.text(x_label, (q33 + q66) / 2.0, f'Moderate C\n(P33–P66)', ha='right', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax1.text(x_label, (q66 + 7.5) / 2.0, f'High C\n(≥ P66={q66:.2f})', ha='right', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Plot 2: Inliers only with confidence band
ax2.scatter(h_inlier, c_inlier, c=rmses[inlier_mask], cmap='viridis',
           alpha=0.7, s=80, edgecolors='black', linewidths=0.5)
ax2.plot(h_range, c_fit, 'r--', linewidth=3, alpha=0.8,
        label=f'Linear Fit (R²={np.corrcoef(h_inlier, c_inlier)[0,1]**2:.3f})')

# Add confidence band (±1 std of residuals)
residuals = c_inlier - (coeffs[1] + coeffs[0] * h_inlier)
std_resid = np.std(residuals)
ax2.fill_between(h_range, c_fit - std_resid, c_fit + std_resid, 
                 alpha=0.2, color='red', label=f'±1σ ({std_resid:.2f})')

ax2.set_xlabel('Normalized H_src', fontsize=13, fontweight='bold')
ax2.set_ylabel('Optimal C Parameter', fontsize=13, fontweight='bold')
ax2.set_title('(Outliers Excluded)',
             fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', 
                          norm=plt.Normalize(vmin=rmses[inlier_mask].min(), 
                                           vmax=rmses[inlier_mask].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2)
cbar.set_label('RMSE (lower is better)', fontsize=11)

plt.tight_layout()

# Save
output_file = os.path.join(project_root, "results", "h_src_vs_c_relationship.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Print usage example
print("\n" + "="*70)
print("USAGE EXAMPLE")
print("="*70)
print("""
from c_predictor import predict_c_from_source

# 
source_pos = (1.2, 1.5, 0.7)
room_dims = (5.5, 7.8, 4.33)
c_opt = predict_c_from_source(source_pos, room_dims)
print(f"Recommended C: {c_opt:.2f}")

""")

plt.show()

