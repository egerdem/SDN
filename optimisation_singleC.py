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
import analysis as an
from rir_calculators import calculate_sdn_rir, rir_normalisation
from functools import partial
from scipy.optimize import minimize_scalar
import plot_room as pp
import matplotlib.pyplot as plt
from copy import deepcopy

# --- Configuration ---
DATA_DIR = "results/paper_data"
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
def compute_spatial_rmse(c_val, room, room_parameters, sdn_config, ref_edcs, receiver_positions, duration, Fs, err_duration_ms):
    """
    Calculates the mean RMSE across all receiver positions for a given 'c' value.
    This is the objective function for the optimizer.
    """

    cut_smpl = int(err_duration_ms * Fs)


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

    for i, (rx, ry) in enumerate(receiver_positions):
        print(f"  Evaluating receiver {i+1}/{num_receivers} at position ({rx:.2f}, {ry:.2f}) with c = {c_scalar:.4f}")
        # 1. Get the pre-computed reference EDC for this receiver
        ref_edc = ref_edcs[i]

        # 2. Calculate the new SDN RIR and EDC with the current 'c' value
        room.set_microphone(rx, ry, room_parameters['mic z']) # Update mic position

        # We need the full room_parameters dict for the calculator
        # room_params = {
        #     'width': room.x, 'depth': room.y, 'height': room.z,
        #     'reflection': 1 - room.wallAttenuation[0]**2,
        #     'mic x': rx, 'mic y': ry, 'mic z': room.micPos.z,
        #     'source x': room.source.srcPos.x, 'source y': room.source.srcPos.y, 'source z': room.source.srcPos.z,
        # }

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
        print(f"    RMSE: {rmse:.6f}")

    mean_rmse = total_rmse / num_receivers
    print(f"  Mean RMSE = {mean_rmse:.6f}")
    return mean_rmse

# --- Main Optimization Loop ---
if __name__ == "__main__":
    for filename in FILES_TO_PROCESS:
        data_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(data_path):
            print(f"\n--- WARNING: File not found, skipping: {data_path} ---")
            continue

        print(f"\n--- Optimizing for file: {filename} ---")

        # 1. Load data
        with np.load(data_path, allow_pickle=True) as data:
            room_parameters = data['room_params'][0]
            source_pos = data['source_pos']
            receiver_positions = data['receiver_positions']
            Fs = int(data['Fs'])
            duration = float(data['duration'])
            # Load all EDCs and get the ones for the reference method
            all_edcs = dict(np.load(data_path, allow_pickle=True))['edcs_RIMPY-neg10']


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

        # 3. Run Optimization
        # Bounds are (1, 7) as requested
        # bounds = [(1.0, 7.0)]
        # x0 = [4.0] # Initial guess, centrally located in bounds
        #
        # minimizer_kwargs = {
        #     "method": "L-BFGS-B",
        #     "bounds": bounds,
        #     "args": (room, base_sdn_config, all_edcs, receiver_positions, duration, Fs, err_duration_ms)
        # }
        #
        # result = basinhopping(compute_spatial_rmse, x0,
        #                       minimizer_kwargs=minimizer_kwargs,
        #                       niter=1, # Number of basin hopping iterations
        #                       disp=True)

        # print("\n--- Optimization Complete ---")
        # print(f"File: {filename}")
        # print(f"Optimal c value: {result.x[0]:.4f}")
        # print(f"Minimum Mean RMSE: {result.fun:.6f}")
        # print("-----------------------------\n")

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

        print(f"Optimal c = {res.x:.3f},  mean RMSE = {res.fun:.6f}")