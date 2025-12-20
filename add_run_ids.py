"""
Add run_id to experiment results

This script loads an experiment results JSON file and adds a 'run_id' key
to each experiment entry, starting from 1 and incrementing by 1.
This makes the results compatible with predict_c_from_proximity.py.
"""

import os
import json

# Define paths
project_root = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(
    project_root,
    
    "results", 
    "paper_data",
    "experiments", 
    # "aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair",
    "aes_fullgrid_8src_diagonal",
    "results.json"
)

# Load the results
print(f"Loading results from: {input_file}")
with open(input_file, 'r') as f:
    results = json.load(f)

print(f"Loaded {len(results)} experiment entries")

# Add run_id to each entry
for i, result in enumerate(results, start=1):
    result['run_id'] = i

print(f"Added run_id from 1 to {len(results)}")

# Save the modified results back to the same file
print(f"Saving modified results to: {input_file}")
with open(input_file, 'w') as f:
    json.dump(results, f, indent=2)

print("Done! Results file updated with run_id keys.")

# Show a few examples
print("\nFirst 3 entries (showing run_id):")
for i in range(min(3, len(results))):
    print(f"  Entry {i}: run_id = {results[i]['run_id']}")
