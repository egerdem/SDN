"""
Optimizes a single source weighting parameter ``c`` for an SDN model using all
available spatial data files for a room.

Where ``optimisation_singleC.py`` computes an optimal ``c`` for each source
position separately, this script finds one global ``c`` that minimises the mean
EDC error across all sources and receivers.
"""

import os
from functools import partial
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize_scalar

import geometry
import analysis as an
from rir_calculators import calculate_sdn_rir, rir_normalisation


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "results/paper_data"
REFERENCE_METHOD = "RIMPY-neg10"
ERROR_DURATION_MS = 50  # compare first 50 ms of the EDC

# Bounds for ``c``.  Can be changed to ``(-3, 7)`` if needed in the future.
BOUNDS = (1.0, 7.0)

FILES_TO_PROCESS = [
    "aes_room_spatial_edc_data_center_source.npz",
    "aes_room_spatial_edc_data_top_middle_source.npz",
    "aes_room_spatial_edc_data_upper_right_source.npz",
    "aes_room_spatial_edc_data_lower_left_source.npz",
]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_dataset(path: str):
    """Load optimisation data from ``path`` and return a dictionary."""
    with np.load(path, allow_pickle=True) as data:
        room_params = data["room_params"][0]
        dataset = {
            "room_params": room_params,
            "source_pos": data["source_pos"],
            "receiver_positions": data["receiver_positions"],
            "Fs": int(data["Fs"]),
            "duration": float(data["duration"]),
            # Each dataset stores pre-computed EDCs for a reference method
            "ref_edcs": data[f"edcs_{REFERENCE_METHOD}"]
        }
    return dataset


def compute_dataset_rmse(c_val: float, dataset: dict, err_duration_ms: int,
                          base_cfg: dict, return_individual: bool = False) -> float:
    """Return mean RMSE over receivers for one source position."""
    Fs = dataset["Fs"]
    duration = dataset["duration"]
    room_params = dataset["room_params"].copy()

    # Set up room and source
    room = geometry.Room(room_params["width"],
                         room_params["depth"],
                         room_params["height"])
    num_samples = int(Fs * duration)
    impulse = geometry.Source.generate_signal("dirac", num_samples)
    room.set_source(*dataset["source_pos"], signal=impulse["signal"], Fs=Fs)
    room.set_microphone(room_params["mic x"], room_params["mic y"],
                        room_params["mic z"])
    room_params["reflection"] = np.sqrt(1 - room_params["absorption"])
    room.wallAttenuation = [room_params["reflection"]] * 6

    cfg = deepcopy(base_cfg)
    cfg["flags"]["source_weighting"] = float(np.squeeze(c_val))
    cfg["label"] = f"SDN-SW-c_{cfg['flags']['source_weighting']:.2f}"

    total_rmse = 0.0
    individual_rmses = []
    receivers = dataset["receiver_positions"]
    ref_edcs = dataset["ref_edcs"]

    for i, (rx, ry) in enumerate(receivers):
        room.set_microphone(rx, ry, room_params["mic z"])

        _, rir_sdn, _, _ = calculate_sdn_rir(room_params, "SDN-Opt", room,
                                             duration, Fs, cfg)
        rir_sdn_normed = rir_normalisation(
            rir_sdn, room, Fs, normalize_to_first_impulse=True
        )["single_rir"]

        edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)
        rmse = an.compute_RMS(
            edc_sdn, ref_edcs[i],
            range=err_duration_ms, Fs=Fs,
            skip_initial_zeros=True,
            normalize_by_active_length=True,
        )
        total_rmse += rmse
        individual_rmses.append(rmse)
        if not return_individual:
            print(f"Receiver {i + 1}/{len(receivers)}: RMSE = {rmse:.6f}")

    mean_rmse = total_rmse / len(receivers)
    if return_individual:
        return mean_rmse, individual_rmses
    return mean_rmse


def compute_total_rmse(c_val: float, datasets: list, err_duration_ms: int,
                        base_cfg: dict) -> float:
    """Objective: mean RMSE across all datasets and receivers."""
    sum_rmse = 0.0
    sum_receivers = 0

    for ds in datasets:
        print(f"Processing dataset with source at {ds['source_pos']}, c = {c_val:.3f}")
        rmse = compute_dataset_rmse(c_val, ds, err_duration_ms, base_cfg)
        n_rx = len(ds["receiver_positions"])
        sum_rmse += rmse * n_rx
        sum_receivers += n_rx
        print(f"  RMSE for this dataset: {rmse:.6f}, ")

    return sum_rmse / sum_receivers


# -----------------------------------------------------------------------------
# Main optimisation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = []
    for fname in FILES_TO_PROCESS:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Warning: file not found: {fpath}")
            continue
        datasets.append(load_dataset(fpath))

    base_cfg = {
        "enabled": True,
        "info": "c_global_optimised",
        "flags": {"specular_source_injection": True},
    }

    obj = partial(
        compute_total_rmse,
        datasets=datasets,
        err_duration_ms=ERROR_DURATION_MS,
        base_cfg=base_cfg,
    )

    result = minimize_scalar(
        obj,
        bounds=BOUNDS,
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 20},
    )

    optimal_c = result.x
    print(f"Optimal c across all sources: {optimal_c:.3f}, mean RMSE = {result.fun:.6f}")

    # Collect individual RMSE results for export
    all_results = {}
    source_names = []
    
    for i, dataset in enumerate(datasets):
        source_name = FILES_TO_PROCESS[i].replace('aes_room_spatial_edc_data_', '').replace('.npz', '').replace('_', ' ').title()
        source_names.append(source_name)
        
        mean_rmse, individual_rmses = compute_dataset_rmse(optimal_c, dataset, ERROR_DURATION_MS, base_cfg, return_individual=True)
        all_results[source_name] = {
            'optimal_c': optimal_c,
            'mean_rmse': mean_rmse,
            'individual_rmses': individual_rmses,
            'source_pos': dataset['source_pos']
        }

    # Export results
    room_info = datasets[0]['room_params']
    print(f"\n--- GLOBAL OPTIMIZATION RESULTS (c={optimal_c:.3f}) ---")
    
    # Console table
    header = f"{'Method':<25}"
    for source_name in source_names:
        header += f" | {source_name}"
    print(header)
    print("-" * len(header))
    
    for i in range(len(datasets[0]['receiver_positions'])):
        rx, ry = datasets[0]['receiver_positions'][i]
        row = f"Receiver {i+1:2d} ({rx:.2f},{ry:.2f})"
        for source_name in source_names:
            rmse = all_results[source_name]['individual_rmses'][i]
            row += f" | {rmse:>15.6f}"
        print(row)
    
    # Export to file
    output_dir = "results/paper_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"global_optimization_results_c_{optimal_c:.3f}_ref_{REFERENCE_METHOD}.txt")
    
    with open(output_path, 'w') as f:
        f.write("--- SDN GLOBAL SOURCE-WEIGHTING OPTIMIZATION RESULTS ---\n")
        f.write(f"Room: {room_info.get('display_name', 'AES Room')} ({room_info['width']}x{room_info['depth']}x{room_info['height']} m)\n")
        f.write(f"Absorption: {room_info['absorption']}, Reference Method: {REFERENCE_METHOD}\n")
        f.write(f"Global Optimal c: {optimal_c:.4f}, Global Mean RMSE: {result.fun:.6f}\n")
        f.write("="*100 + "\n\n")
        
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        for i in range(len(datasets[0]['receiver_positions'])):
            rx, ry = datasets[0]['receiver_positions'][i]
            row = f"Receiver {i+1:2d} ({rx:.2f},{ry:.2f})"
            for source_name in source_names:
                rmse = all_results[source_name]['individual_rmses'][i]
                row += f" | {rmse:>15.6f}"
            f.write(row + "\n")
    
    print(f"\n--- Results exported to: {output_path} ---")
