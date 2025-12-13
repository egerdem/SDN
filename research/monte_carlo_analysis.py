import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the JSON data
file_path = 'monte_carlo_proximity_results.json'

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract metrics
    optimal_c = np.array([d['optimal_c'] for d in data])
    rmse = np.array([d['min_rmse'] for d in data])
    dist_metric = np.array([d['mean_node_dist'] for d in data])
    h_src = np.array([d['h_src'] for d in data])

    # Plotting
    plt.figure(figsize=(14, 6))

    # Subplot 1: Is High C associated with High Error? (Saturation Hypothesis)
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(optimal_c, rmse, c=h_src, cmap='viridis_r', alpha=0.7, edgecolors='k')
    plt.colorbar(scatter, label='Harmonic Mean Dist (Yellow=Corner, Purple=Center)')
    plt.xlabel('Optimal C Value')
    plt.ylabel('Minimum RMSE achieved')
    plt.title('Regime Check: Does Max C = Failure?')
    plt.grid(True, alpha=0.3)

    # Add annotation for saturation
    saturated_mask = optimal_c > 6.8
    # if np.any(saturated_mask):
    #     plt.annotate('Saturation Region\n(Optimizer hits bound)',
    #                  xy=(7, np.mean(rmse[saturated_mask])),
    #                  xytext=(5, np.max(rmse)),
    #                  arrowprops=dict(facecolor='black', shrink=0.05))

    # Subplot 2: The Breakdown of the "Upward Line"
    plt.subplot(1, 2, 2)
    plt.scatter(h_src, optimal_c, c=rmse, cmap='plasma', alpha=0.7, edgecolors='k')
    cbar = plt.colorbar(label='RMSE (Yellow=High Error)')
    plt.xlabel('Harmonic Mean Distance (Source Proximity)')
    plt.ylabel('Optimal C')
    plt.title('Why the "Upward Line" Failed')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('two_regime_analysis.png')
    print("Analysis plot saved to 'two_regime_analysis.png'")
else:
    print("JSON file not found. Please upload 'bound_1to7_monte_carlo_proximity_results.json'")