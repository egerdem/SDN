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
# DATA_FILE = "aes_FULLGRID_center_source.npz"
# DATA_FILE = "aes_FULLGRID_top_middle_source.npz"
# DATA_FILE = "aes_FULLGRID_upper_right_source.npz"
DATA_FILE = "aes_FULLGRID_lower_left_source.npz"
grid_name = DATA_FILE.split('_')[1]
source_name = DATA_FILE.split('_')[2]

# "aes_room_center_source.npz",
# "aes_room_top_middle_source.npz",
# "aes_room_upper_right_source.npz",
# "aes_room_lower_left_source.npz",

NUM_TRIALS = 50  # More trials to see the trend clearly
REFERENCE_METHOD = "RIMPY-neg10"
ERROR_DURATION_MS = 50

# Methods to load from file
PRECOMPUTED_METHODS = [
    'SDN-Test1',      # c=1 (Uniform)
    'SDN-Test_3',     # c=-3
    'SDN-Test_2',     # c=-2
    'SDN-Test2',      # c=2
    'SDN-Test3',      # c=3
    'SDN-Test4',      # c=4
    'SDN-Test5',      # c=5
    'SDN-Test6',      # c=6
    'SDN-Test7'       # c=7
]

def get_c_from_name(name):
    """Extract c value from method name."""
    if name == 'SDN-Test1':
        return 1.0
    
    suffix = name.split('Test')[-1] # "_3", "2", etc.
    if '_' in suffix:
        return -float(suffix.replace('_', ''))
    else:
        return float(suffix)

def get_vector_for_c(c):
    """
    Generate the injection vector for a given weighting c.
    Formula: v[0] = c, others = (5-c)/4. Sum is always 5.
    """
    v = ((5.0 - c) / 4.0) * np.ones(5)
    v[0] = c
    return v

# --- Helper to load data ---
def load_dataset(path):
    with np.load(path, allow_pickle=True) as data:
        result = {
            "room_params": data["room_params"][0],
            "source_pos": data["source_pos"],
            "receiver_positions": data["receiver_positions"],
            "Fs": int(data["Fs"]),
            "duration": float(data["duration"]),
            "ref_edcs": data[f"edcs_{REFERENCE_METHOD}"],
            "precomputed_edcs": {}
        }
        
        # Load EDCs for precomputed methods if they exist
        for method in PRECOMPUTED_METHODS:
            key = f"edcs_{method}"
            if key in data:
                result["precomputed_edcs"][method] = data[key]
            else:
                print(f"Warning: Method {method} not found in dataset")
                
        return result


def generate_sum_constrained_vector(allow_negative=True):
    """
    Generates a random 5-element vector such that sum(v) = 5.0.
    
    Args:
        allow_negative (bool): If True, allows negative elements (constrained by max <= 7).
                             If False, generates only non-negative elements.
    """
    if allow_negative:
        # Purturbation method: Start with uniform and add zero-mean noise
        while True:
            v_base = np.ones(5)
            # Random scale to vary "spikiness" (L2 norm)
            scale = np.random.uniform(0.5, 5.0) 
            noise = np.random.randn(5)
            noise -= np.mean(noise) # Ensure zero sum
            
            v = v_base + scale * noise
            
            # Check constraints
            # 1. Max element <= 7.0
            # 2. Min element > -3.0
            # 3. L2 Norm < sqrt(50) (approx 7.071)
            if np.max(v) <= 7.0 and np.min(v) > -3.0:
                # Sum is naturally 5.0 (5*1 + sum(noise)=0)
                # But float precision might drift, so fast re-normalize
                v = v / np.sum(v) * 5.0 
                
                # Check L2 limit
                if np.linalg.norm(v) < np.sqrt(50):
                    return v
    else:
        # Legacy: Positive-only values
        # 1. Generate random positive values
        mode = np.random.choice(['uniform', 'sparse'])

        if mode == 'uniform':
            v = np.random.rand(5)
        else:
            # Dirichlet distribution with alpha < 1 produces sparse vectors (high L2)
            v = np.random.dirichlet(np.ones(5) * 0.2)

        # 2. Normalize to Sum = 5
        v = v / np.sum(v) * 5.0
        return v


def evaluate_vector(vec, room, room_params, dataset, Fs, duration, base_cfg):
    """
    Evaluates a single injection vector by running the SDN simulation
    and calculating the average RMSE across all receivers.
    """
    # Configure SDN with the current vector
    current_cfg = deepcopy(base_cfg)
    current_cfg["flags"]["source_injection_vector"] = vec.tolist()
    
    # Use ALL receivers for true spatial average
    num_receivers = len(dataset["receiver_positions"])
    receivers = dataset["receiver_positions"]
    ref_edcs = dataset["ref_edcs"]

    total_rmse = 0

    for rx_idx in range(num_receivers):
        rx, ry = receivers[rx_idx][:2]
        # print(f"    ... receiver {rx_idx + 1}/{num_receivers} at ({rx:.2f}, {ry:.2f})")
        room.set_microphone(rx, ry, room_params["mic z"])

        _, rir, _, _ = calculate_sdn_rir(room_params, "Random", room, duration, Fs, current_cfg)

        rir_norm = rir_normalisation(rir, room, Fs)['single_rir']
        edc, _, _ = an.compute_edc(rir_norm, Fs, plot=False)

        rmse = an.compute_RMS(edc, ref_edcs[rx_idx], range=ERROR_DURATION_MS, Fs=Fs,
                              skip_initial_zeros=True, normalize_by_active_length=True)
        total_rmse += rmse

    avg_rmse = total_rmse / num_receivers
    return avg_rmse

def calculate_rmse_from_edcs(edcs, ref_edcs, Fs):
    """Calculate average RMSE for a set of pre-computed EDCs."""
    total_rmse = 0
    num_receivers = len(edcs)
    
    for rx_idx in range(num_receivers):
        rmse = an.compute_RMS(edcs[rx_idx], ref_edcs[rx_idx], range=ERROR_DURATION_MS, Fs=Fs,
                              skip_initial_zeros=True, normalize_by_active_length=True)
        total_rmse += rmse
        
    return total_rmse / num_receivers

def run_experiment():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    data_path = os.path.join(_project_root, "results", "paper_data", DATA_FILE)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    dataset = load_dataset(data_path)

    # Lists to store ALL results for trend fitting
    all_l2 = []
    all_rmse = []
    
    # Store specific groups for plotting
    random_l2 = []
    random_rmse = []
    
    sw_results = [] # List of dicts {c, l2, rmse, is_uniform}

    print(f"--- Starting {NUM_TRIALS} Trials: L2 Norm vs RMSE ---")

    # Base Configuration for Random Trials
    base_cfg = {
        "enabled": True,
        "label": "Random-Trial",
        "flags": {"specular_source_injection": True}
    }

    # Prepare Room for Random Trials
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

    # 1. Process Pre-computed Methods (SW-SDN)
    print("\n--- Processing Pre-computed Methods ---")
    for method, edcs in dataset["precomputed_edcs"].items():
        c = get_c_from_name(method)
        vec = get_vector_for_c(c)
        l2 = np.linalg.norm(vec)
        rmse = calculate_rmse_from_edcs(edcs, dataset["ref_edcs"], Fs)
        
        all_l2.append(l2)
        all_rmse.append(rmse)
        
        sw_results.append({
            'c': c,
            'l2': l2,
            'rmse': rmse,
            'name': method,
            'is_uniform': (c == 1.0)
        })
        print(f"{method} (c={c}): L2={l2:.3f} -> RMSE={rmse:.4f}")

    # 2. Run Random Trials
    print("\n--- Running Random Trials ---")
    for i in range(NUM_TRIALS):
        vec = generate_sum_constrained_vector(allow_negative=True)
        l2_norm = np.linalg.norm(vec)
        avg_rmse = evaluate_vector(vec, room, room_params, dataset, Fs, duration, base_cfg)

        all_l2.append(l2_norm)
        all_rmse.append(avg_rmse)
        random_l2.append(l2_norm)
        random_rmse.append(avg_rmse)

        print(f"Trial {i + 1}/{NUM_TRIALS}: L2={l2_norm:.3f} -> RMSE={avg_rmse:.4f} (Vec={np.round(vec, 2)})", flush=True)

    elapsed_time = time.time() - start_time
    print(f"Experiment took {elapsed_time/60:.2f} minutes.")

    # --- Helper Plot Loop ---
    def make_plot(x_all, y_all, random_x, random_y, sw_data, x_label, file_suffix, use_square=False):
        plt.figure(figsize=(10, 6))
        
        # 1. Plot Random Vectors (Blue dots)
        plt.scatter(random_x, random_y, alpha=0.5, c='blue', s=30, label='Random Vectors')

        # 2. Plot Trend Line (Using ALL data points)
        if len(x_all) > 1:
            # Sort for clean line plotting if needed, though poly1d handles it
            z = np.polyfit(x_all, y_all, 2)
            p = np.poly1d(z)
            xp = np.linspace(min(x_all), max(x_all), 100)
            plt.plot(xp, p(xp), 'r--', label='Global Trend')

        # 3. Plot SW-SDN Methods
        sw_label_added = False
        
        for res in sw_data:
            x_val = res['l2']**2 if use_square else res['l2']
            y_val = res['rmse']
            
            if res['is_uniform']:
                # Black Circle for Uniform (c=1)
                plt.scatter(x_val, y_val, c='black', s=80, marker='o', edgecolors='black', label='Uniform (c=1)', zorder=10)
                plt.text(x_val, y_val + 0.05, f"c={int(res['c'])}", ha='center', fontsize=9, fontweight='bold')
            else:
                # Green Circle for others
                label = 'SW-SDN' if not sw_label_added else None
                plt.scatter(x_val, y_val, c='green', s=80, marker='o', edgecolors='green', label=label, zorder=10)
                if not sw_label_added: sw_label_added = True
                
                # Add text label above check
                plt.text(x_val, y_val + 0.05, f"c={int(res['c'])}", ha='center', fontsize=9)

        plt.xlabel(x_label)
        plt.ylabel('EDC RMSE (dB)')
        plt.title(f'Correlation: Injection Vector {x_label} vs. Error')
        plt.legend()
        plt.grid(True, alpha=0.3)

        out_file = f'l2_norm_correlation_{grid_name}_{source_name}_{file_suffix}.png'
        plt.savefig(out_file)
        print(f"Results plotted to {out_file}")

    # --- Plotting ---
    # Prepare data for plotting (random x/y are already separated)
    
    # 1. L2 Norm Plot
    make_plot(all_l2, all_rmse, random_l2, random_rmse, sw_results, "L2 Norm", "l2")

    # 2. Energy Plot (L2^2)
    all_energy = [x**2 for x in all_l2]
    random_energy = [x**2 for x in random_l2]
    make_plot(all_energy, all_rmse, random_energy, random_rmse, sw_results, "Energy (L2^2)", "energy", use_square=True)

    return (all_l2, all_rmse)


if __name__ == "__main__":
    res = run_experiment()
