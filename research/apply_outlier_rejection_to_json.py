"""
Apply outlier rejection to results.json

This script reads results.json, applies outlier rejection to optimal_c values,
and saves the cleaned version back to the same file (with backup).

Outlier rejection logic:
- If optimal_c hits bounds (1.0 or 7.0 ± 0.005)
- AND c=1 has better (or equal) RMSE
- THEN replace optimal_c with 1.0 and min_rmse with the c=1 RMSE

This ensures all downstream scripts (quick_plot_c_relationship.py, etc.)
use the cleaned data without needing separate logic.

Usage:
    python3 research/apply_outlier_rejection_to_json.py
"""

import os
import sys
import json
import shutil
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configuration
EXPERIMENT_DIR = os.path.join(
    project_root,
    "results",
    "paper_data",
    "experiments",
    "aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair"
)

JSON_PATH = os.path.join(EXPERIMENT_DIR, "results.json")
C_BOUNDS = (1.0, 7.0)


def compute_c1_rmse_from_npz(src_idx, rx_idx, experiment_dir):
    """
    Load RMSE at c=1 from the NPZ file for this source/receiver pair.
    This is needed because results.json doesn't store the c=1 RMSE.
    """
    import numpy as np
    from analysis import analysis as an
    
    # Find the NPZ file for this source
    npz_filename = f"aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair_source_{src_idx+1}.npz"
    npz_path = os.path.join(experiment_dir, npz_filename)
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)
    
    # Get EDCs
    edcs_dict = data['edcs'].item()
    ref_edc = edcs_dict['RIMPY-neg10'][rx_idx]
    sdn_c1_edc = edcs_dict['SDN-Test1'][rx_idx]
    
    # Compute RMSE
    Fs = int(data['Fs'])
    err_duration_ms = 50
    
    rmse_c1 = float(
        an.compute_RMS(
            sdn_c1_edc,
            ref_edc,
            range=err_duration_ms,
            Fs=Fs,
            skip_initial_zeros=True,
            normalize_by_active_length=True,
        )
    )
    
    return rmse_c1


def main():
    print("=" * 80)
    print("Apply Outlier Rejection to results.json")
    print("=" * 80)
    
    if not os.path.exists(JSON_PATH):
        print(f"\n❌ ERROR: JSON file not found: {JSON_PATH}")
        return
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = JSON_PATH.replace(".json", f"_backup_{timestamp}.json")
    shutil.copy2(JSON_PATH, backup_path)
    print(f"\n✅ Created backup: {os.path.basename(backup_path)}")
    
    # Load JSON
    print(f"\nLoading: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} entries")
    
    # Apply outlier rejection
    outlier_count = 0
    eps = 5e-3
    lower, upper = C_BOUNDS
    
    print("\nApplying outlier rejection...")
    
    for entry in data:
        optimal_c = entry.get("optimal_c")
        min_rmse = entry.get("min_rmse")
        src_idx = entry.get("src_idx")
        rx_idx = entry.get("rx_idx")
        
        if optimal_c is None or min_rmse is None:
            continue
        
        # Check if c_opt hits bounds
        hit_upper = abs(optimal_c - upper) <= eps
        hit_lower = abs(optimal_c - lower) <= eps
        
        if hit_upper or hit_lower:
            # Get RMSE at c=1 from NPZ file
            try:
                rmse_c1 = compute_c1_rmse_from_npz(src_idx, rx_idx, EXPERIMENT_DIR)
                
                # If c=1 is better (or equal), use it
                if rmse_c1 <= min_rmse + 1e-6:
                    entry["optimal_c"] = 1.0
                    entry["min_rmse"] = rmse_c1
                    outlier_count += 1
                    
            except Exception as e:
                print(f"  ⚠️  Warning: Could not get c=1 RMSE for src={src_idx}, rx={rx_idx}: {e}")
    
    print(f"  🔧 Applied outlier rejection to {outlier_count} entries")
    
    # Save updated JSON
    print(f"\nSaving updated JSON: {JSON_PATH}")
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✅ Saved {len(data)} entries")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Backup created: {os.path.basename(backup_path)}")
    print(f"Outliers corrected: {outlier_count} / {len(data)} ({100*outlier_count/len(data):.1f}%)")
    print("\n✅ All downstream scripts will now use the cleaned data!")


if __name__ == "__main__":
    main()
