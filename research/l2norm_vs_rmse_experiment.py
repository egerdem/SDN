import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geometry
from analysis import analysis as an
# Use standard calculator as FAST method caches based on wall ID, not arbitrary vectors
from rir_calculators import calculate_sdn_rir, rir_normalisation

# For reproducibility, set random seed
np.random.seed(seed=42)

# --- Config ---
# DATA_FILE = "aes_room_spatial_edc_data_center_source.npz"
# DATA_FILE = "aes_room_spatial_edc_data_upper_right_source.npz"
# DATA_FILE = "aes_room_spatial_edc_data_top_middle_source.npz"
DATA_FILE = "aes_room_spatial_edc_data_lower_left_source.npz"
# "aes_room_spatial_edc_data_lower_left_source.npz",

NUM_TRIALS = 100  # More trials to see the trend clearly
REFERENCE_METHOD = "RIMPY-neg10"
ERROR_DURATION_MS = 50


# --- Helper to load data ---
def load_dataset(path):
    with np.load(path, allow_pickle=True) as data:
        return {
            "room_params": data["room_params"][0],
            "source_pos": data["source_pos"],
            "receiver_positions": data["receiver_positions"],
            "Fs": int(data["Fs"]),
            "duration": float(data["duration"]),
            "ref_edcs": data[f"edcs_{REFERENCE_METHOD}"]
        }


def generate_sum_constrained_vector():
    """
    Generates a random 5-element vector with non-negative elements
    such that sum(v) = 5.0.
    """
    # 1. Generate random positive values
    # Using exponential distribution encourages "spikier" vectors (higher L2 norms)
    # Using uniform distribution encourages "flatter" vectors (lower L2 norms)
    # Let's mix them to cover the whole range of L2 norms [2.23, 5.0]

    mode = np.random.choice(['uniform', 'sparse'])

    if mode == 'uniform':
        v = np.random.rand(5)
    else:
        # Dirichlet distribution with alpha < 1 produces sparse vectors (high L2)
        v = np.random.dirichlet(np.ones(5) * 0.2)

    # 2. Normalize to Sum = 5
    v = v / np.sum(v) * 5.0
    return v


def run_experiment():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    data_path = os.path.join(_project_root, "results", "paper_data", DATA_FILE)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    dataset = load_dataset(data_path)

    results_l2 = []
    results_rmse = []

    print(f"--- Starting {NUM_TRIALS} Trials: L2 Norm vs RMSE ---")

    # Base Configuration
    base_cfg = {
        "enabled": True,
        "label": "Random-Trial",
        "flags": {
            "specular_source_injection": True,
        }
    }

    # Prepare Room
    Fs = dataset["Fs"]
    duration = dataset["duration"]
    room_params = dataset["room_params"]
    room_params["reflection"] = np.sqrt(1 - room_params["absorption"])

    room = geometry.Room(room_params["width"], room_params["depth"], room_params["height"])
    num_samples = int(Fs * duration)
    impulse = geometry.Source.generate_signal("dirac", num_samples)
    room.set_source(*dataset["source_pos"], signal=impulse["signal"], Fs=Fs)
    room.wallAttenuation = [room_params["reflection"]] * 6

    import time
    start_time = time.time()

    # --- Run Trials ---
    for i in range(NUM_TRIALS):
        # 1. Generate Random Vector (Sum=5)
        vec = generate_sum_constrained_vector()

        # 2. Calculate its L2 Norm
        l2_norm = np.linalg.norm(vec)

        # 3. Configure SDN
        current_cfg = deepcopy(base_cfg)
        # sdn_core uses 'source_injection_vector' (list)
        current_cfg["flags"]["source_injection_vector"] = vec.tolist()
        current_cfg["info"] = f"Trial {i} L2={l2_norm:.2f}"

        # 4. Run Simulation (Subset of receivers for speed)
        # We test corners + center to get a robust average
        # Ensure we don't go out of bounds
        num_receivers = len(dataset["receiver_positions"])
        # test_indices = [0, 5, 10, 15] 
        # Pick 4 indices spread out or all if small
        if num_receivers > 4:
             test_indices = np.linspace(0, num_receivers-1, 5, dtype=int).tolist()
        else:
             test_indices = range(num_receivers)
             
        receivers = dataset["receiver_positions"]
        ref_edcs = dataset["ref_edcs"]

        total_rmse = 0

        for rx_idx in test_indices:
            rx, ry = receivers[rx_idx]
            room.set_microphone(rx, ry, room_params["mic z"])

            _, rir, _, _ = calculate_sdn_rir(room_params, "Random", room, duration, Fs, current_cfg)

            rir_norm = rir_normalisation(rir, room, Fs)['single_rir']
            edc, _, _ = an.compute_edc(rir_norm, Fs, plot=False)

            rmse = an.compute_RMS(edc, ref_edcs[rx_idx], range=ERROR_DURATION_MS, Fs=Fs,
                                  skip_initial_zeros=True, normalize_by_active_length=True)
            total_rmse += rmse

        avg_rmse = total_rmse / len(test_indices)

        results_l2.append(l2_norm)
        results_rmse.append(avg_rmse)

        print(f"Trial {i + 1}/{NUM_TRIALS}: L2={l2_norm:.3f} -> RMSE={avg_rmse:.4f} (Vec={np.round(vec, 2)})", flush=True)

    elapsed_time = time.time() - start_time
    print(f"Experiment took {elapsed_time/60:.2f} minutes.")

    # --- Add Reference Points ---
    # 1. Original SDN (Uniform)
    # Vector: [1, 1, 1, 1, 1]
    uniform_vec = np.ones(5)
    uniform_l2 = np.linalg.norm(uniform_vec)  # 2.236
    # You can calculate its RMSE or just plot the L2 line

    # 2. Max Specular (Your Proposal)
    # Vector: [5, 0, 0, 0, 0]
    specular_vec = np.array([5, 0, 0, 0, 0])
    specular_l2 = np.linalg.norm(specular_vec)  # 5.0

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.scatter(results_l2, results_rmse, alpha=0.7, c='blue', label='Random Vectors (Sum=5)')

    # Plot trend line
    if len(results_l2) > 1:
        z = np.polyfit(results_l2, results_rmse, 2)  # 2nd order polynomial fit
        p = np.poly1d(z)
        xp = np.linspace(min(results_l2), max(results_l2), 100)
        plt.plot(xp, p(xp), 'r--', label='Trend')

    # Mark limits
    plt.axvline(x=uniform_l2, color='green', linestyle=':', label='Uniform (c=1)')
    plt.axvline(x=specular_l2, color='purple', linestyle=':', label='Fully Specular (c=5)')

    plt.xlabel('L2 Norm of Injection Vector')
    plt.ylabel('EDC RMSE (dB)')
    plt.title('Correlation: Injection Vector "Sharpness" (L2 Norm) vs. Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_file = 'l2_norm_correlation.png'
    plt.savefig(out_file)
    print(f"Results plotted to {out_file}")


if __name__ == "__main__":
    run_experiment()