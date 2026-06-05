"""
Optimises SDN parameters (source_weighting and/or mic_weighting) by comparing
its EDCs against pre-calculated reference EDCs from spatial data files.

This script supports two-stage optimization:
- Stage 1: Optimize source_weighting (c parameter) with source-side fast method
- Stage 2: Optimize mic_weighting (mic parameter) with mic-side fast method

This script iterates through multiple data files (each representing a different
source position) and finds the optimal parameters for each one.

Now uses FAST caching mechanism (both source-side and mic-side) to avoid 
recalculating RIRs on every iteration.

RELATIONSHIP TO optimisation_globalC.py:
----------------------------------------
This script finds separate optimal parameters for each source position, while
optimisation_globalC.py finds one global parameter set that works across all sources.

IDENTICAL/SIMILAR FUNCTIONS WITH optimisation_globalC.py:
----------------------------------------------------------
1. compute_spatial_rmse() - Core logic is identical to compute_dataset_rmse() 
   in optimisation_globalC.py. Both functions:
   - Take parameter values and compute SDN RIRs for all receiver positions
   - Calculate EDCs and compare against reference EDCs
   - Return mean RMSE (and optionally individual RMSEs)
   - Use FAST method with caching (source-side or mic-side)
   - Main difference: compute_spatial_rmse() takes individual parameters, while 
     compute_dataset_rmse() takes a dataset dict

"""
import os
import sys

# Add project root to path BEFORE other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import basinhopping, minimize_scalar, differential_evolution
import geometry
from analysis import analysis as an
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation, enable_basis_disk_cache, calculate_sdn_rir
from functools import partial
from analysis import plot_room as pp
import matplotlib.pyplot as plt
from copy import deepcopy
from research.data_config import DataConfig

# --- Configuration ---
# Get absolute path to results directory (works regardless of where script is run from)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)  # Go up one level from research/ to project root

# === DATA SOURCE CONFIGURATION ===
# Choose one mode: "legacy" or "experiment"

# Option 1: Legacy mode (flat file structure)
config = DataConfig(mode="legacy", legacy_files=[

    # "journal_FULLGRID_center_source.npz",
    # "journal_FULLGRID_top_middle_source.npz",
    # "journal_FULLGRID_upper_right_source.npz",
    # "journal_FULLGRID_lower_left_source.npz",

    # "cube6_FULLGRID_center_source.npz",
    # "cube6_FULLGRID_top_middle_source.npz",
    # "cube6_FULLGRID_lower_left_source.npz",

    "aes_FULLGRID_center_source.npz",
    "aes_FULLGRID_top_middle_source.npz",
    "aes_FULLGRID_upper_right_source.npz",
    "aes_FULLGRID_lower_left_source.npz",
])

# Option 2: Experiment mode (uncomment to use)
# config = DataConfig(
#     mode="experiment",
#     experiment_name="aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair"
# )

# Get paths from config
DATA_DIR = config.get_data_dir()
FILES_TO_PROCESS = config.get_files()

REFERENCE_METHOD = 'RIMPY-neg10'
err_duration_ms = 50  # 50 ms

# --- Optimizer Selection ---
OPTIMIZER = 'minimize_scalar'  # Options: 'minimize_scalar', 'basin_hopping', 'differential_evolution'
BOUNDS = (1.0, 7.0)  # Bounds for c parameter

# --- Two-Stage Optimization ---
OPTIMIZE_SOURCE_INJECTION = True  # Set to False to skip Stage 1 and use c=1.0 (original SDN)
OPTIMIZE_MIC_WEIGHTING = True  # Set to True to enable Stage 2: mic_weighting optimization
MIC_BOUNDS = (0.0, 6.5)  # Bounds for mic_weighting parameter
SAVE_OPTIMISATION_RESULTS = False

grid_name = "fullgrid"
# --- Objective Function ---
def compute_spatial_rmse(c_val, room, room_parameters, sdn_config, ref_edcs, receiver_positions, duration, Fs, err_duration_ms, source_name, return_individual=False, mic_weighting_val=None):
    """
    Calculates the mean RMSE across all receiver positions for a given 'c' value.
    This is the objective function for the optimizer.
    Uses FAST method with caching (source-side or mic-side depending on optimization stage).
    
    Args:
        mic_weighting_val: Optional mic_weighting value for Stage 2 optimization
            - If None: Uses source-side fast method (use_fast_method=True)
            - If provided: Uses mic-side fast method (use_fast_mic_method=True)
    """
    # Optimizer may pass c as an array, e.g., [3.14]
    c_scalar = np.squeeze(c_val).item()

    # Update the source weighting in the SDN configuration for this evaluation
    cfg = deepcopy(sdn_config)
    cfg['flags']['source_weighting'] = c_scalar.__round__(4)
    
    # Add mic_weighting if provided (Stage 2)
    if mic_weighting_val is not None:
        mic_scalar = np.squeeze(mic_weighting_val).item()
        cfg['flags']['specular_mic_pickup'] = True
        cfg['flags']['mic_weighting'] = mic_scalar
        cfg['label'] = f'SDN-SW-c_{c_scalar:.2f}_mic_{mic_scalar:.2f}'
        # Use mic-side fast method for mic weighting optimization
        cfg['use_fast_mic_method'] = True
        cfg['use_fast_method'] = False  # Disable source-side fast method
    else:
        # Use fast method for source-only optimization
        cfg['use_fast_method'] = True
        cfg['label'] = f'SDN-SW-c_{c_scalar:.2f}'

    total_rmse = 0.0
    num_receivers = len(receiver_positions)
    individual_rmses = []

    for i, pos in enumerate(receiver_positions):
        rx, ry = pos[:2]
        print(f"  Evaluating receiver {i+1}/{num_receivers} at position ({rx:.2f}, {ry:.2f}) with c = {c_scalar:.4f}", end="")
        if mic_weighting_val is not None:
            print(f", mic_weighting = {mic_scalar:.4f}", end="")
            print(f" [DEBUG: cfg source_weighting={cfg['flags'].get('source_weighting', 'MISSING')}, specular_source_injection={cfg['flags'].get('specular_source_injection', 'MISSING')}]")
        else:
            print()
            
        # 1. Get the pre-computed reference EDC for this receiver
        ref_edc = ref_edcs[i]

        # 2. Calculate the new SDN RIR and EDC with the current 'c' value
        room.set_microphone(rx, ry, room_parameters['mic z']) # Update mic position
        
        # Add cache label for this specific receiver
        cfg['cache_label'] = f"{source_name}_rx{i:02d}"
        # print the entire cfg dict (commented out - too verbose)
        # print(cfg)
        
        # Use appropriate RIR calculator based on fast method flags
        if cfg.get('use_fast_method', False) or cfg.get('use_fast_mic_method', False):
            _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_parameters, "SDN-Opt", room, duration, Fs, cfg)
        else:
            _, rir_sdn, _, _ = calculate_sdn_rir(room_parameters, "SDN-Opt", room, duration, Fs, cfg)
            
        rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']

        if  i == 15:
            plot_tr = False
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

# -----------------------------------------------------------------------------
# Reusable Optimization Logic
# -----------------------------------------------------------------------------
def find_optimal_c(room, room_parameters, ref_edcs, receiver_positions, duration, Fs, err_duration_ms=50, source_name="Opt", bounds=BOUNDS):
    """
    Finds the optimal 'c' value for a given room setup.
    Returns: (optimal_c, mean_rmse, individual_rmses)
    """
    # Base SDN config
    base_sdn_config = {
        'enabled': True,
        'info': "c_optimized",
        'flags': {'specular_source_injection': True} # Base for SW-SDN
    }

    # Objective function with fixed arguments
    obj = partial(compute_spatial_rmse,
                  room=room,
                  room_parameters=room_parameters,
                  sdn_config=base_sdn_config,
                  ref_edcs=ref_edcs,
                  receiver_positions=receiver_positions,
                  duration=duration, Fs=Fs,
                  err_duration_ms=err_duration_ms,
                  source_name=source_name)

    # Run Optimization
    # Use minimize_scalar by default as it is most robust for 1D
    res = optimize_minimize_scalar(obj, bounds)
    
    optimal_c = res.x
    mean_rmse = res.fun
    
    # Get details
    _, individual_rmses = compute_spatial_rmse(
        optimal_c, room, room_parameters, base_sdn_config, ref_edcs,
        receiver_positions, duration, Fs, err_duration_ms, source_name, return_individual=True)
        
    return optimal_c, mean_rmse, individual_rmses

# -----------------------------------------------------------------------------
# Optimizer Functions
# -----------------------------------------------------------------------------
def optimize_minimize_scalar(obj_func, bounds):
    """
    1D bounded scalar optimization using Brent's method.
    Fast and reliable for single-parameter optimization.
    """
    # print(f"--- Starting Minimize-Scalar Optimization (bounds={bounds}) ---")
    res = minimize_scalar(obj_func, bounds=bounds, method='bounded',
                          options={'xatol': 1e-3, 'maxiter': 20})
    return res

def optimize_basin_hopping(obj_func, bounds, x0=None):
    """
    Global optimization using basin-hopping with Nelder-Mead local search.
    Good for escaping local minima via random jumps.
    """
    if x0 is None:
        x0 = np.array([(bounds[0] + bounds[1]) / 2])  # Start at midpoint
    
    def print_fun(x, f, accepted):
        print(f"At minimum c={x[0]:.4f} with RMSE={f:.6f}, accepted: {bool(accepted)}")
    
    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "bounds": [bounds],  # List of tuples for basin-hopping
        "options": {"xatol": 1e-2, "fatol": 1e-2, "adaptive": True}
    }
    
    print(f"--- Starting Basin-Hopping Optimization (bounds={bounds}, x0={x0[0]:.2f}) ---")
    result = basinhopping(
        obj_func,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=5,  # Number of basin-hopping iterations
        callback=print_fun,
    )
    return result

def optimize_differential_evolution(obj_func, bounds):
    """
    Global optimization using differential evolution.
    Population-based method, very robust, no initial guess needed.
    """
    print(f"--- Starting Differential Evolution Optimization (bounds={bounds}) ---")
    result = differential_evolution(
        obj_func,
        bounds=[bounds],  # List of tuples for diff evolution
        strategy='best1bin',
        maxiter=50,  # Generations
        popsize=15,  # Population size multiplier
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        polish=True  # Final local search
    )
    return result

# --- Main Optimization Loop ---
if __name__ == "__main__":
    # Set random seed for full reproducibility
    np.random.seed(42)
    
    # Enable disk caching of basis functions for optimization runs
    enable_basis_disk_cache()
    
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

        # Extract source name from filename using config
        source_name = config.extract_source_name_for_display(filename)
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
                'name': room_parameters.get('display_name'),
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
        # Note: Microphone position is set in the loop for each receiver position
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
                      err_duration_ms=err_duration_ms,
                      source_name=source_name)

        # ===== BASELINE: Calculate RMSE with c=1.0 (no optimization) =====
        print(f"\n--- BASELINE: Evaluating with c=1.0 (no optimization) ---")
        baseline_rmse, baseline_individual = compute_spatial_rmse(
            1.0, room, room_parameters, base_sdn_config, all_edcs,
            receiver_positions, duration, Fs, err_duration_ms, source_name, return_individual=True)
        print(f"Baseline RMSE (c=1.0) = {baseline_rmse:.6f}")

        # ===== STAGE 1: Optimize source_weighting (c) =====
        if OPTIMIZE_SOURCE_INJECTION:
            print(f"\n--- STAGE 1: Optimizing Source Injection (c parameter) ---")
            if OPTIMIZER == 'minimize_scalar':
                res = optimize_minimize_scalar(obj, BOUNDS)
            elif OPTIMIZER == 'basin_hopping':
                res = optimize_basin_hopping(obj, BOUNDS)
            elif OPTIMIZER == 'differential_evolution':
                res = optimize_differential_evolution(obj, BOUNDS)
            else:
                raise ValueError(f"Unknown optimizer: {OPTIMIZER}. Choose from: 'minimize_scalar', 'basin_hopping', 'differential_evolution'")

            optimal_c = res.x
            stage1_rmse = res.fun
            print(f"Stage 1 Result: optimal_c = {optimal_c:.4f}, RMSE = {stage1_rmse:.6f}")
            print(f"Improvement over baseline: {baseline_rmse - stage1_rmse:.6f} ({100*(baseline_rmse - stage1_rmse)/baseline_rmse:.2f}%)")

            # Get individual RMSE values from Stage 1
            _, stage1_individual = compute_spatial_rmse(
                optimal_c, room, room_parameters, base_sdn_config, all_edcs,
                receiver_positions, duration, Fs, err_duration_ms, source_name, return_individual=True)
        else:
            # Skip Stage 1: use c=1.0 (original SDN)
            print(f"\n--- STAGE 1: SKIPPED (using c=1.0, original SDN) ---")
            optimal_c = 1.0
            stage1_rmse = baseline_rmse
            stage1_individual = baseline_individual
        
        # ===== STAGE 2: Optimize mic_weighting (optional) =====
        if OPTIMIZE_MIC_WEIGHTING:
            print(f"\n--- STAGE 2: Optimizing Mic Weighting (fixed c={optimal_c:.4f}) ---")
            print(f"Using MIC-SIDE FAST METHOD with caching for efficient optimization")
            
            # Create objective function for mic_weighting optimization
            def mic_objective(mic_val):
                return compute_spatial_rmse(
                    c_val=optimal_c,  # Fixed from Stage 1
                    mic_weighting_val=mic_val,  # Variable
                    room=room,
                    room_parameters=room_parameters,
                    sdn_config=base_sdn_config,
                    ref_edcs=all_edcs,
                    receiver_positions=receiver_positions,
                    duration=duration,
                    Fs=Fs,
                    err_duration_ms=err_duration_ms,
                    source_name=source_name
                )
            
            # Optimize mic_weighting
            # NOTE: First evaluation will compute 2 basis functions (mic_w=0, mic_w=1)
            # for each receiver. Subsequent evaluations use cached basis functions.
            res_mic = optimize_minimize_scalar(mic_objective, MIC_BOUNDS)
            optimal_mic = res_mic.x
            stage2_rmse = res_mic.fun
            
            print(f"Stage 2 Result: optimal_mic_weighting = {optimal_mic:.4f}, RMSE = {stage2_rmse:.6f}")
            print(f"Improvement over Stage 1: {stage1_rmse - stage2_rmse:.6f} ({100*(stage1_rmse - stage2_rmse)/stage1_rmse:.2f}%)")
            print(f"Total improvement over baseline: {baseline_rmse - stage2_rmse:.6f} ({100*(baseline_rmse - stage2_rmse)/baseline_rmse:.2f}%)")
            
            # Get individual RMSE values from Stage 2
            _, stage2_individual = compute_spatial_rmse(
                optimal_c, room, room_parameters, base_sdn_config, all_edcs,
                receiver_positions, duration, Fs, err_duration_ms, source_name, 
                return_individual=True, mic_weighting_val=optimal_mic)
            
            all_results[source_name] = {
                'optimal_c': optimal_c,
                'optimal_mic': optimal_mic,
                'baseline_rmse': baseline_rmse,
                'stage1_rmse': stage1_rmse,
                'stage2_rmse': stage2_rmse,
                'baseline_individual': baseline_individual,
                'stage1_individual': stage1_individual,
                'stage2_individual': stage2_individual,
                'source_pos': source_pos
            }
        else:
            all_results[source_name] = {
                'optimal_c': optimal_c,
                'baseline_rmse': baseline_rmse,
                'stage1_rmse': stage1_rmse,
                'baseline_individual': baseline_individual,
                'stage1_individual': stage1_individual,
                'source_pos': source_pos
            }

    # --- Export Results ---
    if all_results:
        # Print to console
        print("\n" + "="*100)
        print("--- OPTIMIZATION RESULTS SUMMARY ---")
        print("=" * 80)
        print(f"Room: {room_info['name']} ({room_info['dimensions']} m)")
        print(f"Absorption: {room_info['absorption']}, Duration: {room_info['duration']}s, Fs: {room_info['Fs']} Hz")
        print(f"Reference Method: {REFERENCE_METHOD}, Optimization Duration: {err_duration_ms:.0f}ms")
        
        if OPTIMIZE_SOURCE_INJECTION and OPTIMIZE_MIC_WEIGHTING:
            print(f"Two-Stage Optimization: Source Injection (source-side FAST) + Mic Weighting (mic-side FAST)")
        elif OPTIMIZE_SOURCE_INJECTION:
            print(f"Single-Stage Optimization: Source Injection Only (source-side FAST)")
        elif OPTIMIZE_MIC_WEIGHTING:
            print(f"Single-Stage Optimization: Mic Weighting Only (mic-side FAST, c=1.0)")
        else:
            print(f"No Optimization (Baseline only)")
        print("="*100)

        # Build header with source names as columns
        if OPTIMIZE_MIC_WEIGHTING:
            # Two-stage: show both Stage 1 and Stage 2 columns per source
            header = f"{'Method':<25}"
            for source_name in source_names:
                result = all_results[source_name]
                optimal_c = result['optimal_c']
                optimal_mic = result['optimal_mic']
                # Add source name header that spans both columns
                header += f" | {source_name} (c={optimal_c:.2f}, m={optimal_mic:.2f})"
            print(header)
            
            # Sub-header for Stage 1 and Stage 2
            subheader = f"{'':<25}"
            for _ in source_names:
                subheader += f" | {'Stage1':>15} | {'Stage2':>15}"
            print(subheader)
        else:
            # Single-stage: one column per source
            header = f"{'Method':<25}"
            for source_name in source_names:
                result = all_results[source_name]
                optimal_c = result['optimal_c']
                header += f" | {source_name} (c={optimal_c:.2f})"
            print(header)
        
        print("-" * (len(header) + (len(source_names) * 20 if OPTIMIZE_MIC_WEIGHTING else 0)))

        # Receiver rows
        for i in range(room_info['num_receivers']):
            rx, ry, rz = receiver_positions[i]
            row = f"Receiver {i+1:2d} ({rx:.2f},{ry:.2f},{rz:.2f})"
            
            for source_name in source_names:
                result = all_results[source_name]
                
                if OPTIMIZE_MIC_WEIGHTING:
                    # Show both Stage 1 and Stage 2
                    stage1 = result['stage1_individual'][i]
                    stage2 = result['stage2_individual'][i]
                    row += f" | {stage1:>15.6f} | {stage2:>15.6f}"
                else:
                    # Show only Stage 1
                    rmse = result['stage1_individual'][i]
                    row += f" | {rmse:>15.6f}"
            
            print(row)

        # Mean row
        print("-" * (len(header) + (len(source_names) * 20 if OPTIMIZE_MIC_WEIGHTING else 0)))
        mean_row = f"{'Mean RMSE':<25}"
        for source_name in source_names:
            result = all_results[source_name]
            if OPTIMIZE_MIC_WEIGHTING:
                stage1_mean = result['stage1_rmse']
                stage2_mean = result['stage2_rmse']
                mean_row += f" | {stage1_mean:>15.6f} | {stage2_mean:>15.6f}"
            else:
                mean_rmse = result['stage1_rmse']
                mean_row += f" | {mean_rmse:>15.6f}"
        print(mean_row)
        print("="*100)


        # Export to file
        output_dir = DATA_DIR  # Use the same directory as input data
        os.makedirs(output_dir, exist_ok=True)
        room_name_clean = room_info['name'].lower().replace(' ', '_')
        output_filename = f"optimization_{room_name_clean}_{grid_name}_ref_{REFERENCE_METHOD}.txt"
        output_path = os.path.join(output_dir, output_filename)

        if SAVE_OPTIMISATION_RESULTS:
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
                    rx, ry, rz = receiver_positions[i]
                    row = f"Receiver {i+1:2d} ({rx:.2f},{ry:.2f}, {rz:.2f})"
                    for source_name in source_names:
                        rmse = all_results[source_name]['individual_rmses'][i]
                        row += f" | {rmse:>15.6f}"
                    f.write(row + "\n")
                
                f.write("-" * len(header) + "\n")
                f.write(mean_row + "\n")
                f.write("="*100 + "\n")

            print(f"\n--- Results exported to: {output_path} ---")