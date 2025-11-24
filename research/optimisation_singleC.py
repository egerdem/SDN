"""
Optimises a single 'c' (source_weighting) parameter for an SDN model
by comparing its EDCs against pre-calculated reference EDCs from spatial data files.

This script iterates through multiple data files (each representing a different
source position) and finds the optimal 'c' value for each one.
"""
import os
import numpy as np
from scipy.optimize import basinhopping
import geometry
from analysis import analysis as an
from rir_calculators import calculate_sdn_rir, rir_normalisation
from functools import partial
from scipy.optimize import minimize_scalar
from analysis import plot_room as pp
import matplotlib.pyplot as plt
from copy import deepcopy

# --- Configuration ---
# Get absolute path to results directory (works regardless of where script is run from)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)  # Go up one level from research/ to project root
DATA_DIR = os.path.join(_project_root, "results", "paper_data")
REFERENCE_METHOD = 'RIMPY-neg10'
err_duration_ms = 50  # 50 ms

# List of data files to process. Each will be optimized independently.
FILES_TO_PROCESS = [
    "aes_room_spatial_edc_data_center_source.npz",
    "aes_room_spatial_edc_data_top_middle_source.npz",
    "aes_room_spatial_edc_data_upper_right_source.npz",
    "aes_room_spatial_edc_data_lower_left_source.npz",
]

# --- Objective Function ---
def compute_spatial_rmse(c_val, room, room_parameters, sdn_config, ref_edcs, receiver_positions, duration, Fs, err_duration_ms, return_individual=False):
    """
    Calculates the mean RMSE across all receiver positions for a given 'c' value.
    This is the objective function for the optimizer.
    """
    # Optimizer may pass c as an array, e.g., [3.14]
    # c_scalar = c_val[0] if isinstance(c_val, (np.ndarray, list)) else c_val
    c_scalar = np.squeeze(c_val).item()

    # Update the source weighting in the SDN configuration for this evaluation
    cfg = deepcopy(sdn_config)
    cfg['flags']['source_weighting'] = c_scalar

    # sdn_config['flags']['source_weighting'] = c_scalar
    cfg['label'] = f'SDN-SW-c_{c_scalar:.2f}'

    total_rmse = 0.0
    num_receivers = len(receiver_positions)
    individual_rmses = []

    for i, (rx, ry) in enumerate(receiver_positions):
        print(f"  Evaluating receiver {i+1}/{num_receivers} at position ({rx:.2f}, {ry:.2f}) with c = {c_scalar:.4f}")
        # 1. Get the pre-computed reference EDC for this receiver
        ref_edc = ref_edcs[i]

        # 2. Calculate the new SDN RIR and EDC with the current 'c' value
        room.set_microphone(rx, ry, room_parameters['mic z']) # Update mic position

        _, rir_sdn, _, _ = calculate_sdn_rir(room_parameters, "SDN-Opt", room, duration, Fs, cfg)
        rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']

        if i == 14 or i == 15:
            plot_tr = True
        else:
            plot_tr = False

        edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=plot_tr)
        print("rir_sdn_normed:", len(rir_sdn_normed), "edc sdn:", len(edc_sdn), "ref edc:", len(ref_edc))
        # 3. Compute the error and accumulate
        # Ensure sliced EDCs are of the same length for comparison
        rmse = an.compute_RMS(edc_sdn, ref_edc, range=int(err_duration_ms), Fs=Fs,
                 skip_initial_zeros=True,
                 normalize_by_active_length=True)
        total_rmse += rmse
        individual_rmses.append(rmse)
        print(f"    RMSE: {rmse:.6f}")

    mean_rmse = total_rmse / num_receivers
    print(f"  Mean RMSE = {mean_rmse:.6f}")
    
    if return_individual:
        return mean_rmse, individual_rmses
    return mean_rmse

# --- Main Optimization Loop ---
if __name__ == "__main__":
    # Data collection for results export
    all_results = {}
    source_names = []
    room_info = None
    
    for filename in FILES_TO_PROCESS:
        data_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(data_path):
            print(f"\n--- WARNING: File not found, skipping: {data_path} ---")
            continue

        print(f"\n--- Optimizing for file: {filename} ---")

        # Extract source name from filename
        source_name = filename.replace('aes_room_spatial_edc_data_', '').replace('.npz', '').replace('_', ' ').title()
        source_names.append(source_name)

        # 1. Load data
        with np.load(data_path, allow_pickle=True) as data:
            room_parameters = data['room_params'][0]
            source_pos = data['source_pos']
            receiver_positions = data['receiver_positions']
            Fs = int(data['Fs'])
            duration = float(data['duration'])
            # Load all EDCs and get the ones for the reference method
            all_edcs = dict(np.load(data_path, allow_pickle=True))['edcs_RIMPY-neg10']
        
        # Store room info (same for all files)
        if room_info is None:
            room_info = {
                'name': room_parameters.get('display_name', 'Unknown Room'),
                'dimensions': f"{room_parameters['width']}x{room_parameters['depth']}x{room_parameters['height']}",
                'absorption': room_parameters['absorption'],
                'duration': duration,
                'Fs': Fs,
                'num_receivers': len(receiver_positions)
            }

        cut_smpl = int(err_duration_ms * Fs)
        err_duration_ms = err_duration_ms

        # 2. Setup reusable Room object and base SDN config
        room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
        num_samples = int(Fs * duration)
        impulse = geometry.Source.generate_signal('dirac', num_samples)
        room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
        room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z']) # Set initial mic pos
        room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
        room.wallAttenuation = [room_parameters['reflection']] * 6

        base_sdn_config = {
            'enabled': True,
            'info': "c_optimized",
            'flags': {'specular_source_injection': True} # Base for SW-SDN
        }

        obj = partial(compute_spatial_rmse,
                      room=room,
                      room_parameters=room_parameters,
                      sdn_config=base_sdn_config,
                      ref_edcs=all_edcs,
                      receiver_positions=receiver_positions,
                      duration=duration, Fs=Fs,
                      err_duration_ms=err_duration_ms)

        res = minimize_scalar(obj, bounds=(1, 7), method='bounded',
                              options={'xatol': 1e-3, 'maxiter': 10})

        optimal_c = res.x
        print(f"Optimal c = {optimal_c:.3f},  mean RMSE = {res.fun:.6f}")

        # Get individual RMSE values from final evaluation
        final_mean_rmse, individual_rmses = compute_spatial_rmse(
            optimal_c, room, room_parameters, base_sdn_config, all_edcs,
            receiver_positions, duration, Fs, err_duration_ms, return_individual=True)
        
        all_results[source_name] = {
            'optimal_c': optimal_c,
            'mean_rmse': res.fun,
            'individual_rmses': individual_rmses,
            'source_pos': source_pos
        }

    # --- Export Results ---
    if all_results:
        # Print to console
        print("\n" + "="*100)
        print("--- OPTIMIZATION RESULTS SUMMARY ---")
        print(f"Room: {room_info['name']} ({room_info['dimensions']} m)")
        print(f"Absorption: {room_info['absorption']}, Duration: {room_info['duration']}s, Fs: {room_info['Fs']} Hz")
        print(f"Reference Method: {REFERENCE_METHOD}, Optimization Duration: {err_duration_ms:.0f}ms")
        print("="*100)

        # Header
        header = f"{'Method':<25}"
        for source_name in source_names:
            optimal_c = all_results[source_name]['optimal_c']
            header += f" | {source_name} (c={optimal_c:.2f})"
        print(header)
        print("-" * len(header))

        # Receiver rows
        for i in range(room_info['num_receivers']):
            rx, ry = receiver_positions[i]
            row = f"Receiver {i+1:2d} ({rx:.2f},{ry:.2f})"
            for source_name in source_names:
                rmse = all_results[source_name]['individual_rmses'][i]
                row += f" | {rmse:>15.6f}"
            print(row)

        # Mean row
        mean_row = f"{'Mean RMSE':<25}"
        for source_name in source_names:
            mean_rmse = all_results[source_name]['mean_rmse']
            mean_row += f" | {mean_rmse:>15.6f}"
        print("-" * len(header))
        print(mean_row)
        print("="*100)

        # Export to file
        output_dir = DATA_DIR  # Use the same directory as input data
        os.makedirs(output_dir, exist_ok=True)
        room_name_clean = room_info['name'].lower().replace(' ', '_')
        output_filename = f"optimization_results_{room_name_clean}_ref_{REFERENCE_METHOD}.txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as f:
            f.write("--- SDN SOURCE-WEIGHTING OPTIMIZATION RESULTS ---\n")
            f.write(f"Room: {room_info['name']} ({room_info['dimensions']} m)\n")
            f.write(f"Absorption: {room_info['absorption']}, Duration: {room_info['duration']}s, Fs: {room_info['Fs']} Hz\n")
            f.write(f"Reference Method: {REFERENCE_METHOD}\n")
            f.write(f"Optimization Duration: {err_duration_ms:.0f}ms\n")
            f.write(f"Number of Receivers: {room_info['num_receivers']}\n")
            f.write("="*100 + "\n\n")

            # Optimal c values summary
            f.write("OPTIMAL C VALUES BY SOURCE POSITION:\n")
            for source_name in source_names:
                source_pos = all_results[source_name]['source_pos']
                optimal_c = all_results[source_name]['optimal_c']
                mean_rmse = all_results[source_name]['mean_rmse']
                f.write(f"  {source_name}: c = {optimal_c:.4f}, Mean RMSE = {mean_rmse:.6f}, Pos = ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f})\n")
            f.write("\n")

            # Detailed table
            f.write("DETAILED RMSE RESULTS:\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            for i in range(room_info['num_receivers']):
                rx, ry = receiver_positions[i]
                row = f"Receiver {i+1:2d} ({rx:.2f},{ry:.2f})"
                for source_name in source_names:
                    rmse = all_results[source_name]['individual_rmses'][i]
                    row += f" | {rmse:>15.6f}"
                f.write(row + "\n")
            
            f.write("-" * len(header) + "\n")
            f.write(mean_row + "\n")
            f.write("="*100 + "\n")

        print(f"\n--- Results exported to: {output_path} ---")