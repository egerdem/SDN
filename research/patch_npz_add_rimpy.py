"""
Patch existing NPZ files to add RIMPY EDC data from cache.

This script:
1. Reads existing NPZ files that have rirs_RIMPY_neg10 and edcs_RIMPY_neg10
2. Loads corresponding RIMPY RIRs from the rimpy_cache using geometry-based hashing
3. Transforms the data structure to include an 'edcs' dictionary with method keys
4. Overwrites the NPZ files with the corrected format

NO FALLBACKS - will error if cache files don't match, so we know if it's safe.
"""

import os
import sys
import numpy as np
import hashlib
import json

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from analysis import analysis as an

# Configuration
EXPERIMENT_DIR = os.path.join(
    project_root,
    "results",
    "paper_data",
    "experiments",
    # "aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair",
    "aes_fullgrid_8src_diagonal"
)
RIMPY_CACHE_DIR = os.path.join(EXPERIMENT_DIR, "rimpy_cache")


def compute_rimpy_cache_hash(room_dims, src_pos, rx_pos, duration_s, Fs, reflection_sign, rand_dist):
    """
    Replicate the exact hashing logic from monte_carlo_proximity.py _rimpy_cache_paths()
    """
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
    """
    Load RIMPY RIR from cache using geometry-based hash.
    Raises error if not found - NO FALLBACKS.
    """
    # RIMPY-neg10 parameters
    reflection_sign = -1
    rand_dist = 0.1
    
    h = compute_rimpy_cache_hash(
        room_dims, src_pos, rx_pos, duration_s, Fs, reflection_sign, rand_dist
    )
    
    npy_path = os.path.join(cache_dir, f"rimpy_{h}.npy")
    json_path = os.path.join(cache_dir, f"rimpy_{h}.json")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"RIMPY cache file not found: {npy_path}\n"
            f"Hash: {h}\n"
            f"Params: room={room_dims}, src={src_pos}, rx={rx_pos}"
        )
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"RIMPY metadata not found: {json_path}")
    
    # Verify metadata matches
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    # Strict verification - must match exactly
    def close_enough(a, b, tol=1e-9):
        return all(abs(float(x) - float(y)) < tol for x, y in zip(a, b))
    
    if not close_enough(meta["room_dims"], room_dims):
        raise ValueError(f"Room dims mismatch: cache={meta['room_dims']}, expected={room_dims}")
    if not close_enough(meta["src_pos"], src_pos):
        raise ValueError(f"Source pos mismatch: cache={meta['src_pos']}, expected={src_pos}")
    if not close_enough(meta["rx_pos"], rx_pos):
        raise ValueError(f"Receiver pos mismatch: cache={meta['rx_pos']}, expected={rx_pos}")
    if abs(meta["duration_s"] - duration_s) > 1e-6:
        raise ValueError(f"Duration mismatch: cache={meta['duration_s']}, expected={duration_s}")
    if meta["Fs"] != Fs:
        raise ValueError(f"Fs mismatch: cache={meta['Fs']}, expected={Fs}")
    
    # Load RIR
    rir = np.load(npy_path)
    print(f"    ✓ Loaded from cache: rimpy_{h}.npy")
    return rir


def compute_edc_from_rir(rir, Fs):
    """Compute EDC from RIR"""
    edc, _, _ = an.compute_edc(rir, Fs, plot=False)
    return edc


def patch_npz_file(npz_path, cache_dir, dry_run=False):
    """
    Patch a single NPZ file to add proper 'edcs' dictionary structure.
    
    Args:
        npz_path: Path to NPZ file
        cache_dir: Path to RIMPY cache directory
        dry_run: If True, don't actually modify the file
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {os.path.basename(npz_path)}")
    
    # Load existing data
    data = np.load(npz_path, allow_pickle=True)
    
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
    
    n_receivers = len(receiver_positions)
    print(f"  Source: {src_pos}")
    print(f"  Room: {room_dims}")
    print(f"  Receivers: {n_receivers}")
    print(f"  Fs: {Fs}, Duration: {duration}s")
    
    # Check if already has valid 'edcs' dictionary with non-nan RIMPY data
    if 'edcs' in data:
        edcs_dict = data['edcs'].item() if hasattr(data['edcs'], 'item') else data['edcs']
        if 'RIMPY-neg10' in edcs_dict:
            rimpy_data = edcs_dict['RIMPY-neg10']
            # Check if data is valid (not all nan)
            if not np.all(np.isnan(rimpy_data)):
                print("  ✓ Already has valid 'edcs' with RIMPY data - skipping")
                return
            else:
                print("  ⚠️  Has 'edcs' key but RIMPY data is all nan - will patch")
        else:
            print("  ⚠️  Has 'edcs' key but missing RIMPY-neg10 - will patch")
    
    # Load RIMPY RIRs from cache and compute EDCs
    print("  Loading RIMPY data from cache...")
    rimpy_edcs = []
    
    for rx_idx, rx_pos in enumerate(receiver_positions):
        rx_pos_tuple = tuple(float(x) for x in rx_pos)
        
        try:
            rir = load_rimpy_from_cache(
                cache_dir, room_dims, src_pos, rx_pos_tuple, duration, Fs
            )
            edc = compute_edc_from_rir(rir, Fs)
            rimpy_edcs.append(edc)
            
        except Exception as e:
            print(f"  ❌ ERROR loading RIMPY for receiver {rx_idx}: {e}")
            raise
    
    # Pad EDCs to same length
    max_len = max(len(e) for e in rimpy_edcs)
    rimpy_edcs_padded = [
        np.pad(e, (0, max_len - len(e)), 'constant', constant_values=e[-1])
        for e in rimpy_edcs
    ]
    rimpy_edcs_array = np.array(rimpy_edcs_padded)
    
    print(f"  ✓ Loaded all {n_receivers} RIMPY EDCs from cache")
    
    # Get existing SDN-Test1 EDCs
    if 'edcs_SDN_Test1' not in data:
        raise KeyError("Missing 'edcs_SDN_Test1' in NPZ file")
    
    sdn_test1_edcs = data['edcs_SDN_Test1']
    
    # Create new data structure with 'edcs' dictionary
    print("  Creating new data structure...")
    
    new_data = {
        'receiver_positions': receiver_positions,
        'room_params': data['room_params'],
        'source_pos': data['source_pos'],
        'Fs': data['Fs'],
        'duration': data['duration'],
        'method_configs': data['method_configs'],
        # Keep old format for backward compatibility
        'rirs_RIMPY_neg10': data['rirs_RIMPY_neg10'],
        'edcs_RIMPY_neg10': rimpy_edcs_array,  # Update with cache data
        'rirs_SDN_Test1': data['rirs_SDN_Test1'],
        'edcs_SDN_Test1': sdn_test1_edcs,
        # NEW: Add proper 'edcs' dictionary
        'edcs': {
            'RIMPY-neg10': rimpy_edcs_array,
            'SDN-Test1': sdn_test1_edcs,
        }
    }
    
    if dry_run:
        print("  [DRY RUN] Would save updated NPZ file")
        print(f"    - Added 'edcs' dict with keys: {list(new_data['edcs'].keys())}")
        print(f"    - RIMPY-neg10 shape: {rimpy_edcs_array.shape}")
        print(f"    - SDN-Test1 shape: {sdn_test1_edcs.shape}")
    else:
        # Save updated NPZ
        np.savez(npz_path, **new_data)
        print(f"  ✅ Updated NPZ file with 'edcs' dictionary")
        print(f"    - RIMPY-neg10: {rimpy_edcs_array.shape}")
        print(f"    - SDN-Test1: {sdn_test1_edcs.shape}")


def main():
    """Patch all NPZ files in the experiment directory"""
    
    print("=" * 80)
    print("NPZ File Patcher - Add RIMPY EDC data from cache")
    print("=" * 80)
    print(f"\nExperiment dir: {EXPERIMENT_DIR}")
    print(f"RIMPY cache dir: {RIMPY_CACHE_DIR}")
    
    if not os.path.exists(EXPERIMENT_DIR):
        print(f"\n❌ ERROR: Experiment directory not found: {EXPERIMENT_DIR}")
        return
    
    if not os.path.exists(RIMPY_CACHE_DIR):
        print(f"\n❌ ERROR: RIMPY cache directory not found: {RIMPY_CACHE_DIR}")
        return
    
    # Find all NPZ files
    npz_files = [
        f for f in os.listdir(EXPERIMENT_DIR)
        if f.endswith('.npz') and f.startswith('aes_fullgrid')
    ]
    
    if not npz_files:
        print(f"\n❌ No NPZ files found in {EXPERIMENT_DIR}")
        return
    
    npz_files.sort()
    print(f"\nFound {len(npz_files)} NPZ files to process")
    
    # Ask for confirmation
    print("\n" + "=" * 80)
    print("⚠️  WARNING: This will OVERWRITE existing NPZ files!")
    print("=" * 80)
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\nAborted by user.")
        return
    
    # Process each file
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for npz_file in npz_files:
        npz_path = os.path.join(EXPERIMENT_DIR, npz_file)
        try:
            patch_npz_file(npz_path, RIMPY_CACHE_DIR, dry_run=False)
            success_count += 1
        except Exception as e:
            if "Already has 'edcs' key" in str(e):
                skip_count += 1
            else:
                print(f"\n❌ ERROR processing {npz_file}: {e}")
                error_count += 1
                # Don't continue if we hit an error - we want to know immediately
                raise
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(npz_files)}")
    print(f"✅ Successfully patched: {success_count}")
    print(f"⏭️  Skipped (already patched): {skip_count}")
    print(f"❌ Errors: {error_count}")
    
    if error_count == 0:
        print("\n🎉 All files patched successfully!")
    else:
        print(f"\n⚠️  {error_count} file(s) had errors")


if __name__ == "__main__":
    main()
