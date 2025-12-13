
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

import geometry
from analysis import analysis as an
from deprecated import wall_proximity
from optimisation_singleC import find_optimal_c
from rir_calculators import rir_normalisation, calculate_rimpy_rir

# --- Configuration ---
NUM_ROOMS = 4
PAIRS_PER_ROOM = 20
TOTAL_RUNS = NUM_ROOMS * PAIRS_PER_ROOM

# Room Dimensions Ranges (meters)
WIDTH_RANGE = (4.0, 8.0)
DEPTH_RANGE = (4.0, 8.0)
HEIGHT_RANGE = (2.5, 5.0)

# Constants
FS = 44100
DURATION = 1.0  # seconds
ERR_DURATION_MS = 50
ABSORPTION = 0.2 # Fixed absorption for consistency
PADDING = 0.5 # Minimum distance from walls

OUTPUT_FILE = os.path.join(project_root, "results", "monte_carlo_proximity_results.json")

def generate_random_room():
    w = np.random.uniform(*WIDTH_RANGE)
    d = np.random.uniform(*DEPTH_RANGE)
    h = np.random.uniform(*HEIGHT_RANGE)
    return w, d, h

def generate_random_position(w, d, h, padding, mode='uniform'):
    if mode == 'corner_biased' and np.random.rand() < 0.5:
        # 50% chance to force a position deep into a random corner
        # Pick a random corner (0 or Max Dimension)
        x_corner = 0 if np.random.rand() < 0.5 else w
        y_corner = 0 if np.random.rand() < 0.5 else d
        z_corner = 0 if np.random.rand() < 0.5 else h
        
        # Add small random offset (0.1m to 1.5m) ensuring it is inside room
        # We need to move Inwards from the corner
        dx = np.random.uniform(0.1, 1.5)
        dy = np.random.uniform(0.1, 1.5)
        dz = np.random.uniform(0.1, 1.5)
        
        x = x_corner + dx if x_corner == 0 else x_corner - dx
        y = y_corner + dy if y_corner == 0 else y_corner - dy
        z = z_corner + dz if z_corner == 0 else z_corner - dz
        
        # Clamp to be safe (though logic above should handle it if w,d,h > 1.5)
        x = np.clip(x, padding, w - padding)
        y = np.clip(y, padding, d - padding)
        z = np.clip(z, padding, h - padding)
        
        return x, y, z
        
    else:
        # Standard uniform generation
        x = np.random.uniform(padding, w - padding)
        y = np.random.uniform(padding, d - padding)
        z = np.random.uniform(padding, h - padding)
        return x, y, z

def run_monte_carlo():
    print(f"--- Starting Monte Carlo Experiment (N={TOTAL_RUNS}) ---")
    
    # 0. Set Seed for Deterministic "Random" Rooms
    np.random.seed(42)
    
    # Cache Directory
    RIR_CACHE_DIR = os.path.join(project_root, "results", "monte_carlo_rirs")
    os.makedirs(RIR_CACHE_DIR, exist_ok=True)
    
    # Load Existing Results
    existing_results_map = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                data = json.load(f)
                for r in data:
                    existing_results_map[r['run_id']] = r
            print(f"Loaded {len(existing_results_map)} existing results from {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error loading existing results: {e}")

    results = [] # Final list to save
    
    start_time = time.time()
    
    for room_idx in range(NUM_ROOMS):
        # 1. Generate Random Room
        w, d, h = generate_random_room()
        # Round dimensions for cleaner logs
        w, d, h = round(w, 2), round(d, 2), round(h, 2)
        
        room_params = {
            'width': w, 'depth': d, 'height': h,
            'absorption': ABSORPTION, 'reflection': np.sqrt(1 - ABSORPTION),
            'mic z': 1.5, # Placeholder
            # Air absorption parameters (standard conditions)
            'air': {'temperature': 20, 'humidity': 50} 
        }
        
        print(f"\n[Room {room_idx+1}/{NUM_ROOMS}] Dims: {w}x{d}x{h}")
        
        # 2. Generate Pairs
        for pair_idx in range(PAIRS_PER_ROOM):
            run_id = room_idx * PAIRS_PER_ROOM + pair_idx + 1
            
            # --- RESUME LOGIC ---
            if run_id in existing_results_map:
                print(f"  Run {run_id}/{TOTAL_RUNS}: SKIPPING (Result exists)")
                results.append(existing_results_map[run_id])
                
                # Actually, since we set Seed=42 at start, if we skip, we MUST consume the RNG calls
                # OR rely on the fact that we restart the script every time?
                # If we restart script, Seed=42 ensures sequence.
                # If we skip result, we still need to iterate RNG?
                # YES. If we don't call generate_random_position, the Next Room's dimensions will be wrong (shifted).
                _ = generate_random_position(w, d, h, PADDING, mode='corner_biased')
                _ = generate_random_position(w, d, h, PADDING, mode='uniform')
                continue
            
            src_pos = generate_random_position(w, d, h, PADDING, mode='corner_biased')
            mic_pos = generate_random_position(w, d, h, PADDING, mode='uniform') # Generate random Z for mic too
            
            # Update params for rir_calculators
            room_params['source x'], room_params['source y'], room_params['source z'] = src_pos
            room_params['mic x'], room_params['mic y'], room_params['mic z'] = mic_pos
            
            # Volume and Characteristic Length
            vol = w * d * h
            char_len = vol ** (1.0/3.0)
            
            # --- Geometric Analysis ---
            # 1. Wall Proximity (Harmonic Mean)
            src_dists = wall_proximity.get_wall_distances(*src_pos, w, d, h)
            mic_dists = wall_proximity.get_wall_distances(*mic_pos, w, d, h)
            
            h_src = wall_proximity.harmonic_mean(src_dists)
            h_rcv = wall_proximity.harmonic_mean(mic_dists)
            combined_score = (1.0 / h_src) + (1.0 / h_rcv)
            
            # Normalized Metrics (Scale Invariant)
            h_src_norm = h_src / char_len
            
            # 2. Node Distance (Mean 3 Nearest)
            # Create temporary room to calculate nodes
            temp_room = geometry.Room(w, d, h)
            temp_source = geometry.Source(*src_pos, signal=None, Fs=FS)
            temp_room.source = temp_source # Manually attach source
            temp_room.set_microphone(*mic_pos)
            
            node_dists = []
            if temp_room.sdn_nodes_calculated:
                for wall in temp_room.walls.values():
                    np_pos = wall.node_positions
                    dist = temp_room.source.srcPos.getDistance(np_pos)
                    node_dists.append(dist)
                node_dists.sort()
                mean_node_dist = sum(node_dists[:3]) / 3.0
            else:
                mean_node_dist = 0.0 # Should not happen
            
            mean_node_dist_norm = mean_node_dist / char_len
            
            # --- Acoustic Simulation ---
            # 1. Setup Room
            sim_room = geometry.Room(w, d, h)
            impulse = geometry.Source.generate_signal('dirac', int(FS * DURATION))
            sim_room.set_source(*src_pos, signal=impulse['signal'], Fs=FS)
            sim_room.set_microphone(*mic_pos)
            sim_room.wallAttenuation = [room_params['reflection']] * 6
            
            # 2. Calculate Reference & Normalize
            # Check Cache
            rir_filename = f"rimpy_run_{run_id}.npy"
            rir_path = os.path.join(RIR_CACHE_DIR, rir_filename)
            
            loaded_valid_rir = False
            norm_ref = None
            
            if os.path.exists(rir_path):
                # Load from cache
                try:
                    ref_rir = np.load(rir_path)
                    
                    # Robustness Check 1: Energy
                    if np.max(np.abs(ref_rir)) < 1e-9:
                        raise ValueError("Silent RIR")
                        
                    # Robustness Check 2: Normalization (Direct Sound)
                    # We basically try to normalize it. If it fails (divide by zero) or gives NaNs, it's bad.
                    with np.errstate(divide='raise', invalid='raise'):
                        norm_dict = rir_normalisation(ref_rir, sim_room, FS, normalize_to_first_impulse=True)
                        norm_ref = norm_dict['single_rir']
                        
                    if np.any(np.isnan(norm_ref)):
                        raise ValueError("Normalization produced NaNs")
                        
                    loaded_valid_rir = True
                    # print(f"  [Run {run_id}] Loaded cached RIMPY RIR")
                    
                except Exception as e:
                    print(f"  [Run {run_id}] Cached RIR invalid ({e}). Re-calculating.")
                    loaded_valid_rir = False

            if not loaded_valid_rir:
                # Calculate
                # NEW: RIMPY-neg10 (Matches main paper data)
                ref_rir, _ = calculate_rimpy_rir(
                    room_params, DURATION, FS, 
                    reflection_sign=-1, randDist=0.1
                )
                
                # Verify newly calculated one too
                try:
                     with np.errstate(divide='raise', invalid='raise'):
                        norm_dict = rir_normalisation(ref_rir, sim_room, FS, normalize_to_first_impulse=True)
                        norm_ref = norm_dict['single_rir']
                except Exception as e:
                     print(f"  [Run {run_id}] WARNING: Calculated RIR failed normalization too ({e})!")
                     # Fallback? or just let it crash/nan later
                
                # Save to cache
                np.save(rir_path, ref_rir)
                print(f"  [Run {run_id}] Calculated & Cached RIMPY RIR")
            
            # At this point norm_ref is set (or we errored out)
            ref_edc, _, _ = an.compute_edc(norm_ref, FS)
            
            # 3. Optimize SDN
            # find_optimal_c expects `ref_edcs` as a LIST (for multiple receivers).
            # Here we have 1 receiver per pair. So pass `[ref_edc]`.
            receiver_positions = [mic_pos] # List of 1
            
            # Widen bounds to allow finding true optimum even if < 1.0 or > 7.0
            opt_c, min_rmse, _ = find_optimal_c(
                sim_room, room_params, [ref_edc], receiver_positions, 
                DURATION, FS, ERR_DURATION_MS, source_name=f"R{room_idx}_P{pair_idx}",
                bounds=(1, 7.0)
            )
            
            # Store Result
            result = {
                'run_id': run_id,
                'room_dims': (w, d, h),
                'room_vol': vol,
                'char_len': char_len,
                'src_pos': src_pos,
                'mic_pos': mic_pos,
                'h_src': h_src,
                'h_src_norm': h_src_norm,
                'h_rcv': h_rcv,
                'combined_score': combined_score,
                'mean_node_dist': mean_node_dist,
                'mean_node_dist_norm': mean_node_dist_norm,
                'optimal_c': opt_c,
                'min_rmse': min_rmse
            }
            results.append(result)
            
            print(f"  Run {run_id}/{TOTAL_RUNS}: H_src_norm={h_src_norm:.2f}, NodeDist_norm={mean_node_dist_norm:.2f}, C_opt={opt_c:.2f}, RMSE={min_rmse:.4f}")

    # Save Results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nExperiment Complete. Results saved to {OUTPUT_FILE}")
    
    # --- Plotting ---
    plot_results(results)

def plot_results(results):
    h_srcs_norm = [r['h_src_norm'] for r in results]
    combined_scores = [r['combined_score'] for r in results]
    node_dists_norm = [r['mean_node_dist_norm'] for r in results]
    opt_cs = [r['optimal_c'] for r in results]
    rmses = [r['min_rmse'] for r in results]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Normalized Harmonic Mean Source vs Opt C
    plt.subplot(2, 2, 1)
    plt.scatter(h_srcs_norm, opt_cs, alpha=0.6, c=rmses, cmap='viridis')
    plt.xlabel('Normalized H_src (H_src / V^(1/3)) [Lower=Corner]')
    plt.ylabel('Optimal C')
    plt.title('Normalized Source Proximity vs Optimal C')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)
    
    # 2. Combined Score vs Opt C
    plt.subplot(2, 2, 2)
    plt.scatter(combined_scores, opt_cs, alpha=0.6, c=rmses, cmap='viridis')
    plt.xlabel('Combined Proximity Score (1/Hs + 1/Hr)')
    plt.ylabel('Optimal C')
    plt.title('Total Geometric Confinement vs Optimal C')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)
    
    # 3. Normalized Node Distance vs Opt C
    plt.subplot(2, 2, 3)
    plt.scatter(node_dists_norm, opt_cs, alpha=0.6, c=rmses, cmap='viridis')
    plt.xlabel('Normalized Mean 3-Nearest Node Dist (Dist / V^(1/3))')
    plt.ylabel('Optimal C')
    plt.title('Normalized Image Source Density vs Optimal C')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "results", "monte_carlo_correlations_norm.png"))
    print("Plots saved to results/monte_carlo_correlations_norm.png")

if __name__ == "__main__":
    # Ensure results dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    run_monte_carlo()
