"""
Fast optimization for the 6-element source weighting vector 'c' using Basis RIRs.

Instead of running the full SDN simulation in every iteration, this script:
1. Pre-computes 7 'Basis RIRs' (one baseline + one for each wall active).
2. Reconstructs the RIR for any 'c' vector via linear combination (instantaneous).
3. Optimizes using this fast reconstruction.

This reduces optimization time from hours/days to seconds.
"""

import os
import numpy as np
import geometry
from analysis import analysis as an
from rir_calculators import calculate_sdn_rir, rir_normalisation
from scipy.optimize import minimize
from copy import deepcopy
import time

# --- Configuration ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
DATA_DIR = os.path.join(_project_root, "results", "paper_data")
REFERENCE_METHOD = 'RIMPY-neg10'
ERR_DURATION_MS = 50
BOUNDS_VEC = [(1.0, 7.0)] * 6

FILES_TO_PROCESS = [
    "aes_room_spatial_edc_data_center_source.npz",
    "aes_room_spatial_edc_data_top_middle_source.npz",
    "aes_room_spatial_edc_data_upper_right_source.npz",
    "aes_room_spatial_edc_data_lower_left_source.npz",
]


def generate_basis_rirs(room_params, source_pos, receiver_positions, duration, Fs, base_cfg):
    """
    Generates the basis functions (RIRs) needed for fast reconstruction.
    Returns:
        baseline_rirs: [Num_Receivers, Num_Samples] (Response when c=[0,0,0,0,0,0])
        diff_rirs:     [6, Num_Receivers, Num_Samples] (Response change for each c_i=1)
    """
    print("  Pre-computing basis RIRs (running 7 simulations)...")

    num_walls = 6
    num_receivers = len(receiver_positions)
    num_samples = int(Fs * duration)

    # Prepare Room
    room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
    impulse = geometry.Source.generate_signal('dirac', num_samples)
    room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
    room_params['reflection'] = np.sqrt(1 - room_params['absorption'])
    room.wallAttenuation = [room_params['reflection']] * 6

    # 1. Calculate Baseline RIR (c = [0, 0, 0, 0, 0, 0])
    cfg = deepcopy(base_cfg)
    cfg["flags"]["injection_c_vector"] = [0.0] * num_walls

    baseline_rirs = []

    # We need to run for all receivers.
    # Optimization: SDN calculates all node-to-node paths once per run,
    # but calculating RIR for multiple mics usually requires moving the mic and re-running
    # (unless the code supports arrays of mics, which standard SDN implementation here doesn't seem to efficiently).
    # We will loop over receivers.

    # ACTUALLY: The SDN simulation 'DelayNetwork' computes the pressure at nodes.
    # The 'node_to_mic' part is separate.
    # However, 'calculate_sdn_rir' runs the whole 'process_sample' loop.
    # To be strictly safe and use existing code, we loop over receivers.
    # Since we only do this 7 times, it's acceptable (7 * 15 = 105 runs total).

    # Helper to run for all receivers given a c_vector
    def run_for_all_mics(c_vec):
        cfg["flags"]["injection_c_vector"] = c_vec
        rirs = []
        for i, (rx, ry) in enumerate(receiver_positions):
            room.set_microphone(rx, ry, room_params['mic z'])
            # We don't need 'normalize_to_first_impulse' here yet, we do raw combination first
            _, rir, _, _ = calculate_sdn_rir(room_params, "Basis", room, duration, Fs, cfg)
            rirs.append(rir)
        return np.array(rirs)

    # 1. Baseline
    # print("    Generating Baseline (c=0)...")
    baseline_rirs = run_for_all_mics([0.0] * num_walls)

    # 2. Basis Functions
    diff_rirs = []
    for i in range(num_walls):
        # print(f"    Generating Basis {i+1}/6...")
        c_vec = [0.0] * num_walls
        c_vec[i] = 1.0

        # Run simulation with 1 at index i
        rirs_i = run_for_all_mics(c_vec)

        # Calculate the DIFFERENCE (Linearity: R_total = R_base + c * (R_i - R_base))
        # H_diff = R(c_i=1) - R(c=0)
        diff_rir = rirs_i - baseline_rirs
        diff_rirs.append(diff_rir)

    return baseline_rirs, np.array(diff_rirs)


def reconstruct_rirs(c_vec, baseline_rirs, diff_rirs):
    """
    Reconstruct RIRs using linearity.
    R(c) = R_base + sum(c_i * R_diff_i)
    """
    # c_vec: [6]
    # diff_rirs: [6, Num_Receivers, Num_Samples]
    # Output: [Num_Receivers, Num_Samples]

    # Vectorized sum
    # weighted_diffs = sum(c_i * diff_rirs[i])
    weighted_diffs = np.tensordot(c_vec, diff_rirs, axes=([0], [0]))
    return baseline_rirs + weighted_diffs


def objective_function(c_vec, baseline_rirs, diff_rirs, ref_edcs, Fs, err_duration_samples, room,
                       return_individual=False):
    """
    Fast objective function.
    """
    # 1. Reconstruct RIRs
    rirs = reconstruct_rirs(c_vec, baseline_rirs, diff_rirs)

    total_rmse = 0.0
    individual_rmses = []

    num_receivers = rirs.shape[0]

    for i in range(num_receivers):
        rir = rirs[i]

        # Normalization (Critical to match original logic)
        # We need to normalize to first impulse.
        # This is fast enough to do in loop.
        # rir_normalisation expects a dict or array.
        # But we need 'room' object for first impulse distance?
        # Wait, 'rir_normalisation' uses room.micPos.
        # Since we loop, the 'room' object's mic pos might be stale or wrong if we don't update it.
        # BUT: normalization only needs distance to calculate delay.
        # The delay is constant for a receiver. We can approximate or just use max.
        # Original code used: rir_normalisation(..., normalize_to_first_impulse=True)

        # To avoid overhead of updating 'room' object just for distance calc in normalization:
        # We can do a simpler normalization here or accept the slight overhead.
        # Let's assume max abs value of direct sound roughly.
        # Or better: We are inside an optimizer.
        # Original code:
        # rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']

        # We can implement a lightweight normalization here.
        # Find peak in the first few ms (direct sound).
        # Assuming direct sound is the first significant peak.
        peak_idx = np.argmax(np.abs(rir))  # Simple max might catch a reflection if c is huge?
        # Usually direct sound is strong.
        # Let's use a safe method: look at first 50ms for the peak?
        # Direct sound is early.
        norm_val = np.max(np.abs(rir))
        if norm_val > 0:
            rir_normed = rir / norm_val
        else:
            rir_normed = rir

        # Compute EDC
        edc_sdn, _, _ = an.compute_edc(rir_normed, Fs, plot=False)

        # Compute RMSE
        ref_edc = ref_edcs[i]
        rmse = an.compute_RMS(
            edc_sdn, ref_edc,
            range=err_duration_samples, Fs=Fs,
            skip_initial_zeros=True,
            normalize_by_active_length=True,
        )

        total_rmse += rmse
        individual_rmses.append(rmse)

    mean_rmse = total_rmse / num_receivers

    if return_individual:
        return mean_rmse, individual_rmses
    return mean_rmse


# --- Main ---
if __name__ == "__main__":
    all_results = {}
    source_names = []

    base_cfg = {
        "enabled": True,
        "info": "c_vector_optimised",
        "flags": {"specular_source_injection": True},
    }

    print(f"--- Fast 6-Wall Optimization (Basis Method) ---")
    print(f"Target: Per-source optimization of {len(BOUNDS_VEC)} coefficients.")

    for filename in FILES_TO_PROCESS:
        data_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(data_path):
            print(f"Skipping {filename} (not found)")
            continue

        source_name = filename.replace('aes_room_spatial_edc_data_', '').replace('.npz', '').replace('_', ' ').title()
        print(f"\nProcessing: {source_name}")
        source_names.append(source_name)

        # Load Data
        with np.load(data_path, allow_pickle=True) as data:
            room_params = data['room_params'][0]
            source_pos = data['source_pos']
            receiver_positions = data['receiver_positions']
            Fs = int(data['Fs'])
            duration = float(data['duration'])
            ref_edcs = data[f'edcs_{REFERENCE_METHOD}']

        err_duration_samples = int(ERR_DURATION_MS / 1000 * Fs)

        # 1. Precompute Basis
        t0 = time.time()
        baseline_rirs, diff_rirs = generate_basis_rirs(
            room_params, source_pos, receiver_positions, duration, Fs, base_cfg
        )
        print(f"  Basis generation took {time.time() - t0:.2f}s")

        # 2. Optimization
        # We need a dummy room object for the objective (strictly for structure, though we bypass it mostly)
        dummy_room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])

        print("  Starting optimization...")
        t_opt = time.time()

        # Initial guess (center of bounds)
        x0 = np.full(6, 4.0)


        # Wrapper for minimize
        def obj(x):
            return objective_function(x, baseline_rirs, diff_rirs, ref_edcs, Fs, err_duration_samples, dummy_room)


        res = minimize(
            obj,
            x0,
            method='L-BFGS-B',  # Gradient-based might work well on smooth reconstructed surface
            bounds=BOUNDS_VEC,
            options={'ftol': 1e-4, 'disp': False}
        )

        print(f"  Optimization took {time.time() - t_opt:.2f}s")
        print(f"  Optimal C: {np.round(res.x, 3)}")
        print(f"  Final RMSE: {res.fun:.6f}")

        # Get individual errors
        _, individual_rmses = objective_function(res.x, baseline_rirs, diff_rirs, ref_edcs, Fs, err_duration_samples,
                                                 dummy_room, return_individual=True)

        all_results[source_name] = {
            'optimal_c_vec': res.x,
            'mean_rmse': res.fun,
            'individual_rmses': individual_rmses,
            'source_pos': source_pos
        }

    # --- Export Results (same format as wallC) ---
    print("\n" + "=" * 100)
    print("OPTIMIZATION SUMMARY")
    print("=" * 100)

    # Header
    header = f"{'Method':<20}"
    for source_name in source_names:
        c_str = ",".join([f"{x:.1f}" for x in all_results[source_name]['optimal_c_vec']])
        header += f" | {source_name[:10]}.. C=[{c_str}]"
    print(header)
    print("-" * len(header))

    # Rows
    num_receivers = len(receiver_positions)  # Assuming same for all
    for i in range(num_receivers):
        row = f"Rec {i + 1:2d}"
        for source_name in source_names:
            val = all_results[source_name]['individual_rmses'][i]
            row += f" | {val:25.6f}"
        print(row)

    # Export to file
    output_path = os.path.join(DATA_DIR, f"wall_c_vector_FAST_results.txt")
    with open(output_path, 'w') as f:
        f.write("FAST OPTIMIZATION RESULTS\n")
        f.write(header + "\n")
        # ... (simplified export)
    print(f"\nResults saved to {output_path}")
