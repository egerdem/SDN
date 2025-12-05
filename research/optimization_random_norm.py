"""
Optimizes 6 scalar gains for 6 fixed RANDOM injection vectors.
Tests the "Shape vs. Energy" hypothesis.

1. Generates 6 random 5-element vectors (one for each wall).
2. Normalizes them to sum to 5.0 (preserving first-order reflections).
3. Optimizes 6 scalar gains (one per wall) that scale these vectors.
"""

import os
import sys
import numpy as np
from functools import partial
from copy import deepcopy

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geometry
from analysis import analysis as an
from rir_calculators import calculate_sdn_rir, rir_normalisation
from research.optimisation_singleC import optimize_minimize_scalar, optimize_basin_hopping, optimize_differential_evolution

# Reuse configuration from optimisation_wallC
from research.optimisation_wallC import load_dataset, DATA_DIR, REFERENCE_METHOD, ERROR_DURATION_MS

# --- Configuration ---
BOUNDS = [(2, 25)] * 6  # Bounds for the scalar gains

def generate_random_vectors(seed=None):
    """
    Generates 6 random 5-element vectors, normalized to unit L2 norm.
    Returns: List of 6 numpy arrays.
    """
    if seed is not None:
        np.random.seed(seed)
    
    vectors = []
    for _ in range(6):
        # Generate random vector (positive values only)
        v = np.random.rand(5) 
        
        # Normalize so sum is 5.0 (Constraint: sum of elements = K-1 = 5)
        total = np.sum(v)
        if total > 0:
            v = v * (5.0 / total)
        vectors.append(v)
    
    return vectors

def compute_dataset_rmse_random_norm(gains_vec, random_vectors, dataset, err_duration_ms, base_cfg):
    """
    Computes RMSE for a given set of 6 gains, applied to the fixed random vectors.
    """
    Fs = dataset["Fs"]
    duration = dataset["duration"]
    room_params = dataset["room_params"].copy()

    # Set up room
    room = geometry.Room(room_params["width"], room_params["depth"], room_params["height"])
    num_samples = int(Fs * duration)
    impulse = geometry.Source.generate_signal("dirac", num_samples)
    room.set_source(*dataset["source_pos"], signal=impulse["signal"], Fs=Fs)
    room_params["reflection"] = np.sqrt(1 - room_params["absorption"])
    room.wallAttenuation = [room_params["reflection"]] * 6

    # Construct the full 6x5 injection matrix
    # source_injection_vector = [gain_i * vector_i]
    full_injection_matrix = []
    for i in range(6):
        gain = gains_vec[i]
        vec = random_vectors[i]
        scaled_vec = (vec * gain).tolist()
        full_injection_matrix.append(scaled_vec)

    cfg = deepcopy(base_cfg)
    cfg["flags"]["source_injection_vector"] = full_injection_matrix
    # We do NOT set node_weighting_vector or source_weighting, as source_injection_vector takes precedence
    
    # Label
    gains_str = "-".join([f"{g:.2f}" for g in gains_vec])
    cfg["label"] = f"SDN-RandNorm-Gains_{gains_str}"

    total_rmse = 0.0
    receivers = dataset["receiver_positions"]
    ref_edcs = dataset["ref_edcs"]

    for i, (rx, ry) in enumerate(receivers):
        room.set_microphone(rx, ry, room_params["mic z"])
        
        # Note: FAST method might not support list-of-lists injection vector yet.
        # Using standard calculation for safety.
        _, rir_sdn, _, _ = calculate_sdn_rir(room_params, "SDN-Opt", room, duration, Fs, cfg)
        
        rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']
        edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)
        
        rmse = an.compute_RMS(
            edc_sdn, ref_edcs[i],
            range=err_duration_ms, Fs=Fs,
            skip_initial_zeros=True,
            normalize_by_active_length=True,
        )
        total_rmse += rmse

    return total_rmse / len(receivers)

def objective_function(gains_vec, random_vectors, datasets, err_duration_ms, base_cfg):
    """Mean RMSE across all datasets."""
    total_rmse = 0.0
    total_count = 0
    
    individual_rmses = []
    for ds in datasets:
        rmse = compute_dataset_rmse_random_norm(gains_vec, random_vectors, ds, err_duration_ms, base_cfg)
        individual_rmses.append(rmse)
        total_rmse += rmse
        total_count += 1
    
    mean_rmse = total_rmse / total_count
    # Print individual RMSEs as a compact list
    rmses_str = "[" + ", ".join([f"{r:.3f}" for r in individual_rmses]) + "]"
    print(f"  Eval: Gains={np.round(gains_vec, 2)}, Mean={mean_rmse:.4f}, Indiv={rmses_str}", flush=True)
    return mean_rmse

if __name__ == "__main__":
    print("--- Random Norm Optimization (Suggestion 1) ---", flush=True)

    # List of data files to process. Each will be optimized independently.
    FILES_TO_PROCESS = [
        "aes_quarter_center_source.npz",
        # "aes_room_spatial_edc_data_top_middle_source.npz",
        # "aes_room_spatial_edc_data_upper_right_source.npz",
        # "aes_room_spatial_edc_data_lower_left_source.npz",
    ]

    # 1. Load Data
    datasets = []
    for fname in FILES_TO_PROCESS:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            datasets.append(load_dataset(fpath))
            print(f"Loaded {fname}", flush=True)
    
    if not datasets:
        print("No datasets found.")
        sys.exit(1)

    # 2. Generate Fixed Random Vectors
    # Use a fixed seed for reproducibility of this run
    SEED = 42
    print(f"Generating 6 random vectors (Seed={SEED}, Sum=5.0)...", flush=True)
    random_vectors = generate_random_vectors(seed=SEED)
    for i, v in enumerate(random_vectors):
        print(f"  Wall {i}: {np.round(v, 3)} (Sum={np.sum(v):.2f}, norm_squared={np.linalg.norm(v):.2f})", flush=True)

    # 3. Setup Optimization
    base_cfg = {
        "enabled": True,
        "info": "random_norm_opt",
        "flags": {"specular_source_injection": True},
    }

    OPTIMIZER = 'differential_evolution'  # Choose from: 'minimize_scalar', 'basin_hopping', 'differential_evolution'

    obj = partial(
        objective_function,
        random_vectors=random_vectors,
        datasets=datasets,
        err_duration_ms=ERROR_DURATION_MS,
        base_cfg=base_cfg
    )

    # 4. Run Optimization (Differential Evolution is best for this multi-modal space)
    print("\nStarting Differential Evolution...")
    from scipy.optimize import differential_evolution

    import time
    start_time = time.time()

    res = differential_evolution(
        obj,
        bounds=BOUNDS,
        strategy='best1bin',
        maxiter=10,
        popsize=10,
        disp=True,
        polish=True
    )

    elapsed_time = time.time() - start_time
    print(f"\nOptimization took {elapsed_time/60:.2f} minutes.")
    # if OPTIMIZER == 'minimize_scalar':
    #     res = optimize_minimize_scalar(obj, BOUNDS)
    # elif OPTIMIZER == 'basin_hopping':
    #     res = optimize_basin_hopping(obj, BOUNDS)
    # elif OPTIMIZER == 'differential_evolution':
    #     res = optimize_differential_evolution(obj, BOUNDS)
    # else:
    #     raise ValueError(
    #         f"Unknown optimizer: {OPTIMIZER}. Choose from: 'minimize_scalar', 'basin_hopping', 'differential_evolution'")

    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print(f"Final Mean RMSE: {res.fun:.6f}")
    print("Optimal Gains:")
    print(np.round(res.x, 4))
    print("="*50)
    
    # Print effective injection vectors (Gain * Vector)
    print("\nEffective Injection Vectors (Gain * RandomVector):")
    for i in range(6):
        eff_vec = res.x[i] * random_vectors[i]
        print(f"  Wall {i}: {np.round(eff_vec, 3)}")
