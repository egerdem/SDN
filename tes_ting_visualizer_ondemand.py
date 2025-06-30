"""
Test script to run the visualizer with on-demand metrics calculation.

This allows testing changes to EDC/NED calculation algorithms without 
reloading all the data or recomputing metrics in the manager.
"""

import random
from sdn_manager_load_sims import ExperimentLoaderManager
from sdn_experiment_visualizer import ExperimentVisualizer

# Setup paths
results_dir = 'results'

# Load a single specific batch project - change to your desired project
batch_manager = ExperimentLoaderManager(
    results_dir=results_dir, 
    is_batch_manager=True,
    disable_unified_rooms = True,  # Disable unified rooms
    skip_metrics=True,  # Skip metrics calculation
    project_names="aes_quartergrid_new"  # Change this to your desired project
)

# Create visualizer with the loaded experiments
visualizer = ExperimentVisualizer(batch_manager)

# Generate random port number to avoid conflicts
port_no = random.randint(1000, 9999)

# Launch the visualizer with on-demand calculation enabled
print(f"\n=== LAUNCHING VISUALIZER WITH ON-DEMAND CALCULATION ===")
print(f"Any changes to EDC/NED calculation algorithms in:")
print(f"  - analysis.py")
print(f"  - EchoDensity.py")
print(f"will be visible without reloading data.\n")

# Launch the visualizer with on-demand calculation
visualizer.show(port=port_no, use_on_demand_calculations=True) 