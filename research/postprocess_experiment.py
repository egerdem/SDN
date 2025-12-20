"""
Unified Post-Processing Script for Monte Carlo Experiments

This script combines all post-processing steps into one:
1. Add run_id to results.json
2. Patch NPZ files with RIMPY EDC data from cache
3. Populate CSV with optimal_c from results.json (with outlier rejection)
4. Apply outlier rejection to results.json

Usage:
    python3 research/postprocess_experiment.py --experiment aes_fullgrid_8src_diagonal

Or edit EXPERIMENT_NAME below and run:
    python3 research/postprocess_experiment.py
"""

import os
import sys
import json
import numpy as np
import hashlib
import shutil
from datetime import datetime
import argparse

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from analysis import analysis as an

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default experiment name (can be overridden with --experiment flag)
EXPERIMENT_NAME = "aes_fullgrid_8src_diagonal"

# Outlier rejection settings
APPLY_OUTLIER_REJECTION = True
C_BOUNDS = (1.0, 7.0)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_rimpy_cache_hash(room_dims, src_pos, rx_pos, duration_s, Fs, reflection_sign, rand_dist):
    """Replicate the exact hashing logic from monte_carlo_proximity.py"""
    payload = {
        "room_dims": [float(x) for x in room_dims],
        "src_pos": [float(x) for x in src_pos],
        "rx_pos": [float(x) for x in rx_pos],
        "duration_s": float(duration_s),
        "Fs": int(Fs),
        "reflection_sign": int(reflection_sign),
        "rand_dist": float(rand_dist),
    }
    key = json.dumps(payload, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(key).hexdigest()[:12]
    return h


def load_rimpy_from_cache(cache_dir, room_dims, src_pos, rx_pos, duration_s, Fs):
    """Load RIMPY RIR from cache using geometry-based hash"""
    reflection_sign = -1
    rand_dist = 0.1
    
    h = compute_rimpy_cache_hash(
        room_dims, src_pos, rx_pos, duration_s, Fs, reflection_sign, rand_dist
    )
    
    npy_path = os.path.join(cache_dir, f"rimpy_{h}.npy")
    json_path = os.path.join(cache_dir, f"rimpy_{h}.json")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"RIMPY cache file not found: {npy_path}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"RIMPY metadata not found: {json_path}")
    
    # Verify metadata matches
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    def close_enough(a, b, tol=1e-9):
        return all(abs(float(x) - float(y)) < tol for x, y in zip(a, b))
    
    if not close_enough(meta["room_dims"], room_dims):
        raise ValueError(f"Room dims mismatch")
    if not close_enough(meta["src_pos"], src_pos):
        raise ValueError(f"Source pos mismatch")
    if not close_enough(meta["rx_pos"], rx_pos):
        raise ValueError(f"Receiver pos mismatch")
    
    rir = np.load(npy_path)
    return rir


# ============================================================================
# STEP 1: ADD RUN IDS
# ============================================================================

def step1_add_run_ids(experiment_dir):
    """Add run_id to each entry in results.json"""
    print("\n" + "="*80)
    print("STEP 1: Adding run_id to results.json")
    print("="*80)
    
    json_path = os.path.join(experiment_dir, "results.json")
    
    if not os.path.exists(json_path):
        print(f"❌ ERROR: results.json not found at {json_path}")
        return False
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} entries")
    
    # Check if run_id already exists
    if results and 'run_id' in results[0]:
        print("⏭️  run_id already exists, skipping")
        return True
    
    # Add run_id
    for i, result in enumerate(results, start=1):
        result['run_id'] = i
    
    # Save
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Added run_id from 1 to {len(results)}")
    return True


# ============================================================================
# STEP 2: PATCH NPZ FILES
# ============================================================================

def step2_patch_npz_files(experiment_dir):
    """Patch NPZ files with RIMPY EDC data from cache"""
    print("\n" + "="*80)
    print("STEP 2: Patching NPZ files with RIMPY data")
    print("="*80)
    
    cache_dir = os.path.join(experiment_dir, "rimpy_cache")
    
    if not os.path.exists(cache_dir):
        print(f"❌ ERROR: RIMPY cache not found at {cache_dir}")
        return False
    
    # Find NPZ files
    npz_files = [
        f for f in os.listdir(experiment_dir)
        if f.endswith('.npz') and 'fullgrid' in f
    ]
    
    if not npz_files:
        print("❌ No NPZ files found")
        return False
    
    npz_files.sort()
    print(f"Found {len(npz_files)} NPZ files")
    
    success_count = 0
    skip_count = 0
    
    for npz_file in npz_files:
        npz_path = os.path.join(experiment_dir, npz_file)
        
        # Load existing data
        data = np.load(npz_path, allow_pickle=True)
        
        # Check if already has valid RIMPY data
        if 'edcs' in data:
            edcs_dict = data['edcs'].item() if hasattr(data['edcs'], 'item') else data['edcs']
            if 'RIMPY-neg10' in edcs_dict:
                rimpy_data = edcs_dict['RIMPY-neg10']
                # Check if data is valid (not all nan)
                if not np.all(np.isnan(rimpy_data)):
                    skip_count += 1
                    continue
                else:
                    print(f"  ⚠️  RIMPY data is nan, will patch")
        
        print(f"\nPatching: {npz_file}")
        
        # Extract metadata
        Fs = int(data["Fs"])
        duration = float(data["duration"])
        room_params = data["room_params"].item()
        src_pos = tuple(float(x) for x in data["source_pos"])
        receiver_positions = data["receiver_positions"]
        
        room_dims = (
            float(room_params["width"]),
            float(room_params["depth"]),
            float(room_params["height"])
        )
        
        # Load RIMPY EDCs from cache
        rimpy_edcs = []
        for rx_pos in receiver_positions:
            rx_pos_tuple = tuple(float(x) for x in rx_pos)
            rir = load_rimpy_from_cache(
                cache_dir, room_dims, src_pos, rx_pos_tuple, duration, Fs
            )
            edc, _, _ = an.compute_edc(rir, Fs, plot=False)
            rimpy_edcs.append(edc)
        
        # Pad to same length
        max_len = max(len(e) for e in rimpy_edcs)
        rimpy_edcs_array = np.array([
            np.pad(e, (0, max_len - len(e)), 'constant', constant_values=e[-1])
            for e in rimpy_edcs
        ])
        
        # Create new data structure
        new_data = {
            'receiver_positions': receiver_positions,
            'room_params': data['room_params'],
            'source_pos': data['source_pos'],
            'Fs': data['Fs'],
            'duration': data['duration'],
            'method_configs': data['method_configs'],
            'rirs_RIMPY_neg10': data['rirs_RIMPY_neg10'],
            'edcs_RIMPY_neg10': rimpy_edcs_array,
            'rirs_SDN_Test1': data['rirs_SDN_Test1'],
            'edcs_SDN_Test1': data['edcs_SDN_Test1'],
            'edcs': {
                'RIMPY-neg10': rimpy_edcs_array,
                'SDN-Test1': data['edcs_SDN_Test1'],
            }
        }
        
        # Save
        np.savez(npz_path, **new_data)
        success_count += 1
    
    print(f"\n✅ Patched: {success_count}, Skipped: {skip_count}")
    return True


# ============================================================================
# STEP 3: APPLY OUTLIER REJECTION TO JSON
# ============================================================================

def step3_apply_outlier_rejection(experiment_dir):
    """Apply outlier rejection to results.json"""
    if not APPLY_OUTLIER_REJECTION:
        print("\n⏭️  STEP 3: Outlier rejection disabled, skipping")
        return True
    
    print("\n" + "="*80)
    print("STEP 3: Applying outlier rejection to results.json")
    print("="*80)
    
    json_path = os.path.join(experiment_dir, "results.json")
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = json_path.replace(".json", f"_backup_{timestamp}.json")
    shutil.copy2(json_path, backup_path)
    print(f"✅ Created backup: {os.path.basename(backup_path)}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Load NPZ files to get RMSE at c=1
    npz_files = {}
    for f in os.listdir(experiment_dir):
        if f.endswith('.npz') and 'fullgrid' in f:
            npz_files[f] = np.load(os.path.join(experiment_dir, f), allow_pickle=True)
    
    outlier_count = 0
    eps = 5e-3
    lower, upper = C_BOUNDS
    
    for entry in data:
        optimal_c = entry.get("optimal_c")
        min_rmse = entry.get("min_rmse")
        src_idx = entry.get("src_idx")
        rx_idx = entry.get("rx_idx")
        
        if optimal_c is None or min_rmse is None:
            continue
        
        # Check if hits bounds
        hit_upper = abs(optimal_c - upper) <= eps
        hit_lower = abs(optimal_c - lower) <= eps
        
        if hit_upper or hit_lower:
            # Get RMSE at c=1 from NPZ
            npz_filename = f"aes_fullgrid_8src_diagonal_source_{src_idx+1}.npz"
            if npz_filename in npz_files:
                npz_data = npz_files[npz_filename]
                edcs_dict = npz_data['edcs'].item()
                ref_edc = edcs_dict['RIMPY-neg10'][rx_idx]
                sdn_c1_edc = edcs_dict['SDN-Test1'][rx_idx]
                
                Fs = int(npz_data['Fs'])
                rmse_c1 = float(an.compute_RMS(
                    sdn_c1_edc, ref_edc,
                    range=50, Fs=Fs,
                    skip_initial_zeros=True,
                    normalize_by_active_length=True
                ))
                
                if rmse_c1 <= min_rmse + 1e-6:
                    entry["optimal_c"] = 1.0
                    entry["min_rmse"] = rmse_c1
                    outlier_count += 1
    
    # Save
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Applied outlier rejection to {outlier_count} entries")
    return True


# ============================================================================
# STEP 4: POPULATE CSV FROM JSON
# ============================================================================

def step4_populate_csv_from_json(experiment_dir):
    """Populate aes_pairwise_c_analysis.csv with optimal_c from results.json"""
    print("\n" + "="*80)
    print("STEP 4: Populating CSV with optimal_c from results.json")
    print("="*80)
    
    json_path = os.path.join(experiment_dir, "results.json")
    csv_path = os.path.join(experiment_dir, "aes_pairwise_c_analysis.csv")
    
    if not os.path.exists(csv_path):
        print(f"⏭️  CSV file not found, skipping (will be created by aes_pairwise_c_analysis.py)")
        return True
    
    # Load JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Create lookup
    lookup = {}
    for entry in json_data:
        src_idx = entry.get("src_idx")
        rx_idx = entry.get("rx_idx")
        optimal_c = entry.get("optimal_c")
        min_rmse = entry.get("min_rmse")
        
        if src_idx is not None and rx_idx is not None:
            lookup[(src_idx, rx_idx)] = {
                "optimal_c": optimal_c,
                "min_rmse": min_rmse
            }
    
    # Load CSV
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    # Check if CSV has the required columns
    if 'src_idx' not in fieldnames or 'rx_idx' not in fieldnames:
        print(f"⚠️  CSV missing src_idx or rx_idx columns, skipping")
        return True
    
    # Update rows
    updated_count = 0
    for row in rows:
        try:
            src_idx = int(row['src_idx'])
            rx_idx = int(row['rx_idx'])
            
            if (src_idx, rx_idx) in lookup:
                data = lookup[(src_idx, rx_idx)]
                row['c_opt_pair'] = data['optimal_c']
                row['rmse_opt_pair'] = data['min_rmse']
                updated_count += 1
        except (KeyError, ValueError) as e:
            print(f"⚠️  Skipping row due to error: {e}")
            continue
    
    # Save CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Updated {updated_count} rows in CSV")
    return True


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Post-process Monte Carlo experiment")
    parser.add_argument(
        "--experiment",
        type=str,
        default=EXPERIMENT_NAME,
        help=f"Experiment name (default: {EXPERIMENT_NAME})"
    )
    args = parser.parse_args()
    
    experiment_dir = os.path.join(
        project_root,
        "results",
        "paper_data",
        "experiments",
        args.experiment
    )
    
    print("="*80)
    print("UNIFIED POST-PROCESSING SCRIPT")
    print("="*80)
    print(f"Experiment: {args.experiment}")
    print(f"Directory: {experiment_dir}")
    
    if not os.path.exists(experiment_dir):
        print(f"\n❌ ERROR: Experiment directory not found")
        return
    
    # Run all steps
    if not step1_add_run_ids(experiment_dir):
        print("\n❌ Step 1 failed, aborting")
        return
    
    if not step2_patch_npz_files(experiment_dir):
        print("\n❌ Step 2 failed, aborting")
        return
    
    if not step3_apply_outlier_rejection(experiment_dir):
        print("\n❌ Step 3 failed, aborting")
        return
    
    if not step4_populate_csv_from_json(experiment_dir):
        print("\n❌ Step 4 failed, aborting")
        return
    
    print("\n" + "="*80)
    print("🎉 ALL POST-PROCESSING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run: python3 research/populate_csv_from_json.py")
    print("  2. Run: python3 research/predict_c_from_proximity.py")
    print("  3. Run: python3 research/aes_pairwise_c_analysis.py")


if __name__ == "__main__":
    main()
