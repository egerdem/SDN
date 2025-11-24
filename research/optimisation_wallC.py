"""
Optimises a 6-element source weighting vector ``c`` for an SDN model
by minimising the EDC RMSE averaged over all available spatial data
files.

Each element of the vector corresponds to the ``injection_c_vector``
parameter used during specular source injection.  All six values share
the same optimisation bounds of 1 to 7 (future work may extend this to
-3 to 7).
"""

import os
from functools import partial
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

import geometry
from analysis import analysis as an
from rir_calculators import calculate_sdn_rir, calculate_sdn_rir_fast, rir_normalisation


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Get absolute path to results directory (works regardless of where script is run from)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)  # Go up one level from research/ to project root
DATA_DIR = os.path.join(_project_root, "results", "paper_data")
REFERENCE_METHOD = "RIMPY-neg10"
ERROR_DURATION_MS = 50  # compare first 50 ms of the EDC

# Bounds for each c parameter
BOUNDS_VEC = [(1.0, 7.0)] * 6

# Limit number of receivers for testing (set to None to use all)
MAX_RECEIVERS = 16  # For trial runs, limit to first 2 receivers

FILES_TO_PROCESS = [
    "aes_room_spatial_edc_data_center_source.npz",
    # "aes_room_spatial_edc_data_lower_left_source.npz",
    # "aes_room_spatial_edc_data_top_middle_source.npz",
    # "aes_room_spatial_edc_data_upper_right_source.npz",
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
            "ref_edcs": data[f"edcs_{REFERENCE_METHOD}"]
        }
    return dataset


def compute_dataset_rmse(c_vec: np.ndarray, dataset: dict, err_duration_ms: int,
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
    c_list = [float(v) for v in np.squeeze(c_vec)]
    cfg["flags"]["injection_c_vector"] = c_list
    cfg["label"] = "SDN-SW-cvec_" + "-".join(f"{v:.2f}" for v in c_list)

    total_rmse = 0.0
    individual_rmses = []

    receivers = dataset["receiver_positions"]
    ref_edcs = dataset["ref_edcs"]
    
    # Limit receivers for testing if MAX_RECEIVERS is set
    if MAX_RECEIVERS is not None and len(receivers) > MAX_RECEIVERS:
        receivers = receivers[:MAX_RECEIVERS]
        ref_edcs = ref_edcs[:MAX_RECEIVERS]
        print(f"  [Trial Mode] Limiting to first {MAX_RECEIVERS} receivers")

    for i, (rx, ry) in enumerate(receivers):
        room.set_microphone(rx, ry,  room_params["mic z"])

        # Use FAST method for instant reconstruction (basis functions cached per receiver)
        _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_params, "SDN-Opt", room,
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
        print(f"Receiver {i + 1}/{len(receivers)}, ({rx:.2f},{ry:.2f}): RMSE = {rmse:.6f}")

    mean_rmse = total_rmse / len(receivers)
    print(f"Mean RMSE for source at {dataset['source_pos']}: {mean_rmse:.6f}")
    if return_individual:
        return mean_rmse, individual_rmses

    return mean_rmse


def compute_total_rmse(c_vec: np.ndarray, datasets: list, err_duration_ms: int,
                        base_cfg: dict) -> float:
    """Objective: mean RMSE across all datasets and receivers."""
    sum_rmse = 0.0
    sum_receivers = 0

    for ds in datasets:
        print(f"Processing dataset with source at {ds['source_pos']}, c = {c_vec}")
        rmse = compute_dataset_rmse(c_vec, ds, err_duration_ms, base_cfg)
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

    if not datasets:
        raise RuntimeError("No data files loaded for optimisation.")

    base_cfg = {
        "enabled": True,
        "info": "c_vector_optimised",
        "flags": {"specular_source_injection": True},
    }

    obj = partial(
        compute_total_rmse,
        datasets=datasets,
        err_duration_ms=ERROR_DURATION_MS,
        base_cfg=base_cfg,
    )

    x0 = np.full(6, 1.24)

    # print("--- Starting Nelder-Mead Optimization ---")
    # result = minimize(
    #     obj,
    #     x0,
    #     method="Nelder-Mead",  # <<< Change the method here
    #     bounds=BOUNDS_VEC,
    #     # A generous tolerance might be needed for such a long function
    #     options={"maxiter": 10, "xatol": 1e-3, "fatol": 1e-3, "adaptive": True},
    # )

    # best_x0_from_stage_1 = [4.6895261,  6.29991025, 1.08560566, 6.97723387, 1.10307365, 1.9357747]
    # print("--- Starting Stage 2: Precise local search ---")
    # result = minimize(
    #     obj,  # The full objective function with all data
    #     best_x0_from_stage_1,
    #     method="Nelder-Mead",
    #     bounds=BOUNDS_VEC,
    #     options={"xatol": 1e-1, "fatol": 1e-1, "adaptive": True},  # Use a slightly tighter tolerance
    # )


    # # Define a callback to see progress
    from scipy.optimize import basinhopping
    def print_fun(x, f, accepted):
        print(f"At minimum {x} with value {f}, accepted: {bool(accepted)}")


    # Configure the local minimizer to be used for each "hop"
    minimizer_kwargs = {
        "method": "Nelder-Mead",  # Using Nelder-Mead for local search is robust
        "bounds": BOUNDS_VEC,
        "options": {"xatol": 1e-1, "fatol": 1e-1, "adaptive": True}
    }

    print("--- Starting Basin-Hopping Optimization ---")
    result = basinhopping(
        obj,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=5,  # Number of basin-hopping iterations
        callback=print_fun,
    )

    optimal_c_vec = result.x
    print("Optimal c vector:", np.round(optimal_c_vec, 3))
    # print(f"Mean RMSE: {result.fun:.6f}")

    # Collect individual RMSE results for export
    all_results = {}
    source_names = []
    
    for i, dataset in enumerate(datasets):
        source_name = FILES_TO_PROCESS[i].replace('aes_room_spatial_edc_data_', '').replace('.npz', '').replace('_', ' ').title()
        source_names.append(source_name)
        
        mean_rmse, individual_rmses = compute_dataset_rmse(optimal_c_vec, dataset, ERROR_DURATION_MS, base_cfg, return_individual=True)
        all_results[source_name] = {
            'optimal_c_vec': optimal_c_vec,
            'mean_rmse': mean_rmse,
            'individual_rmses': individual_rmses,
            'source_pos': dataset['source_pos']
        }

    # Export results
    room_info = datasets[0]['room_params']
    c_vec_str = "[" + ",".join(f"{v:.2f}" for v in optimal_c_vec) + "]"
    print(f"\n--- WALL C-VECTOR OPTIMIZATION RESULTS {c_vec_str} ---")
    
    # Console table
    header = f"{'Method':<25}"
    for source_name in source_names:
        header += f" | {source_name}"
    print(header)
    print("-" * len(header))
    
    # Determine number of receivers to display (respect MAX_RECEIVERS limit)
    num_receivers = len(datasets[0]['receiver_positions'])
    if MAX_RECEIVERS is not None:
        num_receivers = min(num_receivers, MAX_RECEIVERS)
    
    for i in range(num_receivers):
        rx, ry = datasets[0]['receiver_positions'][i]
        row = f"Receiver {i+1:2d} ({rx:.2f},{ry:.2f})"
        for source_name in source_names:
            rmse = all_results[source_name]['individual_rmses'][i]
            row += f" | {rmse:>15.6f}"
        print(row)
    
    # Export to file
    output_dir = DATA_DIR  # Use the same directory as input data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"wall_c_vector_optimization_results_ref_{REFERENCE_METHOD}.txt")
    
    with open(output_path, 'w') as f:
        f.write("--- SDN WALL C-VECTOR OPTIMIZATION RESULTS ---\n")
        f.write(f"Room: {room_info.get('display_name', 'AES Room')} ({room_info['width']}x{room_info['depth']}x{room_info['height']} m)\n")
        f.write(f"Absorption: {room_info['absorption']}, Reference Method: {REFERENCE_METHOD}\n")
        f.write(f"Optimal c vector: {c_vec_str}, Global Mean RMSE: {result.fun:.6f}\n")
        f.write("="*100 + "\n\n")
        
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # Determine number of receivers to display (respect MAX_RECEIVERS limit)
        num_receivers = len(datasets[0]['receiver_positions'])
        if MAX_RECEIVERS is not None:
            num_receivers = min(num_receivers, MAX_RECEIVERS)
        
        for i in range(num_receivers):
            rx, ry = datasets[0]['receiver_positions'][i]
            row = f"Receiver {i+1:2d} ({rx:.2f},{ry:.2f})"
            for source_name in source_names:
                rmse = all_results[source_name]['individual_rmses'][i]
                row += f" | {rmse:>15.6f}"
            f.write(row + "\n")
    
    print(f"\n--- Results exported to: {output_path} ---")


