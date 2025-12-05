import numpy as np
import os
import json
from typing import Dict, List, Tuple
import geometry
from analysis import analysis as an
from rir_calculators import calculate_pra_rir, calculate_sdn_rir, calculate_ho_sdn_rir, calculate_rimpy_rir, rir_normalisation, calculate_sdn_rir_fast
from analysis.spatial_analysis import generate_receiver_grid_old, generate_source_positions, generate_full_receiver_grid

def calculate_and_save_data(room_params: dict, source_pos: Tuple[float, float, float],
                              receiver_positions: List[Tuple[float, float]],
                              duration: float, Fs: int,
                              method_configs: Dict,
                              output_path: str,
                              update_mode):
    """
    Calculate RIRs, EDCs, and NEDs. Can update an existing data file.
    If update_mode is True, it loads existing data and only calculates missing
    data for methods marked as 'enabled' in the script's config.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Data Loading for Update Mode ---
    all_rirs = {}
    all_edcs = {}
    all_neds = {}
    all_rt60s = {}
    final_method_configs = method_configs.copy()

    if update_mode and os.path.exists(output_path):
        print(f"--- UPDATE MODE: Loading existing data from {output_path} ---")
        with np.load(output_path, allow_pickle=True) as loaded_data:
            # Pre-populate results with existing data
            for key in loaded_data.keys():
                if key.startswith('rirs_'):
                    method = key.replace('rirs_', '')
                    all_rirs[method] = loaded_data[key]
                elif key.startswith('edcs_'):
                    method = key.replace('edcs_', '')
                    all_edcs[method] = loaded_data[key]
                elif key.startswith('neds_'):
                    method = key.replace('neds_', '')
                    all_neds[method] = loaded_data[key]
                elif key.startswith('rt60s_'):
                    method = key.replace('rt60s_', '')
                    all_rt60s[method] = loaded_data[key]

            # Load the existing method configurations and merge them
            if 'method_configs' in loaded_data:
                existing_configs = loaded_data['method_configs'][0]
                existing_configs.update(final_method_configs)
                final_method_configs = existing_configs

    # --- Determine which methods to process ---
    methods_to_process = [m for m, c in method_configs.items() if c.get('enabled', False)]

    if not methods_to_process:
        print("No methods are enabled in the configuration. Nothing to do.")
        return

    print(f"--- Checking and processing enabled methods: {methods_to_process} ---")

    # --- Simulation Setup ---
    room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
    num_samples = int(Fs * duration)
    impulse = geometry.Source.generate_signal('dirac', num_samples)
    room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
    room.wallAttenuation = [np.sqrt(1 - room_params['absorption'])] * 6
    room_parameters = room_params.copy()
    room_parameters['reflection'] = np.sqrt(1 - room_params['absorption'])

    # --- Main Processing Loop ---
    any_new_data_calculated = False
    for method in methods_to_process:
        print(f"\n--- Processing Method: {method} ---")
        config = method_configs[method]

        # --- Step 1: Check and Calculate RIRs ---
        # Check if method exists AND if it should be replaced
        method_exists = method in all_rirs
        force_replace = method in METHODS_TO_REPLACE
        
        if not method_exists or force_replace:
            if force_replace and method_exists:
                print(f"  FORCE REPLACING existing data for '{method}'...")
            else:
                print(f"  RIRs not found for '{method}'. Calculating...")
            any_new_data_calculated = True
            current_method_rirs = []
            for i, pos in enumerate(receiver_positions):
                rx, ry = pos[:2]
                print(f"    ... receiver {i+1}/{len(receiver_positions)} at ({rx:.2f}, {ry:.2f})")

                # Set microphone on the main room object for this iteration.
                # This is used by `calculate_sdn_rir` and `rir_normalisation`.
                room.set_microphone(rx, ry, room_params['mic z'])

                current_params_for_calc = room_parameters.copy()
                current_params_for_calc.update({
                    'mic x': rx,
                    'mic y': ry,
                    'source x': source_pos[0],
                    'source y': source_pos[1],
                    'source z': source_pos[2],
                })

                calculator = config.get('calculator')

                if calculator == 'pra':
                    rir, _ = calculate_pra_rir(current_params_for_calc, duration, Fs, **config.get('params', {}))
                elif calculator == 'rimpy':
                    rir, _ = calculate_rimpy_rir(current_params_for_calc, duration, Fs, **config.get('params', {}))
                elif method.startswith('SDN-'):
                    print("Calculating SDN...")
                    _, rir, _, _ = calculate_sdn_rir_fast(current_params_for_calc, method, room, duration, Fs, config)
                elif method.startswith('HO-SDN'):
                    source_signal = config.get('source_signal', 'dirac')
                    order = config.get('order')
                    rir, _ = calculate_ho_sdn_rir(current_params_for_calc, Fs, duration, source_signal, order=order)
                else:
                    print(f"    Warning: Unknown or unsupported method {method}, skipping.")
                    continue

                if np.max(np.abs(rir)) > 0:
                    # Normalize to the direct sound impulse instead of the global max.
                    # This provides a consistent, physically meaningful anchor point.
                    normalized_rir_dict = rir_normalisation(rir, room, Fs, normalize_to_first_impulse=True)
                    rir = next(iter(normalized_rir_dict.values()))

                current_method_rirs.append(rir)
                print(f"  {len(rir)}, Calculated {len(current_method_rirs)} RIRs for method '{method}'.")
            all_rirs[method] = np.array(current_method_rirs)

        else:
            print(f"  RIRs for '{method}' already exist. Skipping calculation.")

        # --- Step 2: Check and Calculate EDCs ---
        edc_exists = method in all_edcs
        if (not edc_exists or force_replace) and method in all_rirs:
            if force_replace and edc_exists:
                print(f"  FORCE REPLACING existing EDCs for '{method}'...")
            else:
                print(f"  EDCs not found for '{method}'. Calculating...")
            any_new_data_calculated = True
            edcs_for_method = []
            for rir in all_rirs[method]:
                edc, _, _ = an.compute_edc(rir, Fs, plot=False)
                edcs_for_method.append(edc)

            # Pad all EDCs to the same length to handle inconsistencies from compute_edc's tail-trimming
            if edcs_for_method:
                max_len = max(len(e) for e in edcs_for_method)
                padded_edcs = [np.pad(e, (0, max_len - len(e)), 'constant', constant_values=e[-1]) for e in edcs_for_method]
                all_edcs[method] = np.array(padded_edcs)

        elif edc_exists:
            print(f"  EDCs for '{method}' already exist. Skipping calculation.")

        # --- Step 3: Check and Calculate NEDs (BYPASSED) ---
        # if method not in all_neds and method in all_rirs:
        #     print(f"  NEDs not found for '{method}'. Calculating...")
        #     any_new_data_calculated = True
        #     neds_for_method = []
        #     for rir in all_rirs[method]:
        #         ned_profile = ned.echoDensityProfile(rir, fs=Fs)
        #         neds_for_method.append(ned_profile)
        #     all_neds[method] = np.array(neds_for_method)
        # elif method in all_neds:
        #     print(f"  NEDs for '{method}' already exist. Skipping calculation.")

        # --- Step 4: Check and Calculate RT60s ---
        rt60_exists = method in all_rt60s
        if (not rt60_exists or force_replace) and method in all_rirs:
            if force_replace and rt60_exists:
                print(f"  FORCE REPLACING existing RT60s for '{method}'...")
            else:
                print(f"  RT60s not found for '{method}'. Calculating...")
            any_new_data_calculated = True
            rt60_values = []
            for rir in all_rirs[method]:
                rt60 = an.calculate_rt60_from_rir(rir, Fs, plot=False)
                rt60_values.append(rt60 if rt60 is not None else np.nan)
            all_rt60s[method] = np.array(rt60_values)
        elif rt60_exists:
            print(f"  RT60s for '{method}' already exist. Skipping calculation.")


    # --- Data Saving ---
    if any_new_data_calculated:
        print(f"\nSaving updated data ({len(all_rirs)} methods) to: {output_path}")
        save_dict = {
            'receiver_positions': np.array(receiver_positions),
            'room_params': np.array([room_params]),
            'source_pos': np.array(source_pos),
            'Fs': np.array(Fs),
            'duration': np.array(duration),
            'method_configs': np.array([final_method_configs])
        }

        for method_key, rirs_data in all_rirs.items():
            save_dict[f'rirs_{method_key}'] = rirs_data
            if method_key in all_edcs:
                save_dict[f'edcs_{method_key}'] = all_edcs[method_key]
            if method_key in all_neds:
                save_dict[f'neds_{method_key}'] = all_neds[method_key]
            if method_key in all_rt60s:
                save_dict[f'rt60s_{method_key}'] = all_rt60s[method_key]

        np.savez(output_path, **save_dict)
        print("--- Save complete ---")
    else:
        print("\n--- No new data was calculated. File is already up-to-date. ---")


if __name__ == "__main__":
    import sys
    # Ensure root directory is in path to import experiment_configs
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import experiment_configs as exp_config

    # --- CONFIGURATION PARAMETERS ---
    UPDATE_MODE = True  # If True, loads existing data and adds new methods
    
    # Force replace specific methods even if they already exist in the file
    # Useful when you need to regenerate data with updated code/parameters
    METHODS_TO_REPLACE = [
    #     'SDN-Test2.998',  # Regenerate using Standard (non-Fast) SDN method
    ]
    # Set to True to load the existing data file and only run/replace
    # the methods that are marked 'enabled' in this script.
    UPDATE_EXISTING_FILE = False # Set to False to always calculate everything from scratch.
    use_grid = True # Set to True to use a grid of receiver positions, False for single position
    
    # --- GRID SELECTION ---
    # Options: "full", "quarter"
    GRID_SELECTION = "full" 

    # Files to process when PROCESS_MULTIPLE_SOURCES is True
    # HOW TO USE:
    # This list determines which files creation will be ATTEMPTED.
    # To generate a new file (e.g., for Center Source), UNCOMMENT its name below.
    # The script acts as a "whitelist": it will only generate files listed here.
    FILES_TO_PROCESS = [
        # "aes_room_center_source.npz",
        # "aes_room_top_middle_source.npz",
        # "aes_room_upper_right_source.npz",
        # "aes_room_lower_left_source.npz",
        
        # New FULL GRID files (GRID_SELECTION = "full"):
        "aes_FULLGRID_center_source.npz",
        # "aes_FULLGRID_top_middle_source.npz",
        # "aes_FULLGRID_upper_right_source.npz",
        # "aes_FULLGRID_lower_left_source.npz",
        # "aes_FULLGRID_lower_left_source.npz",
        
        # Legacy files (GRID_SELECTION = "quarter"):
        # "aes_room_lower_left_source.npz",
    ]
    
    # Set to True to process multiple sources, False for single source from active_room
    PROCESS_MULTIPLE_SOURCES = True 

    # Use active room from config
    active_room = exp_config.active_room

    # Merge method configurations
    method_configs = {}
    method_configs.update(exp_config.ism_methods)
    method_configs.update(exp_config.sdn_tests)
    method_configs.update(exp_config.ho_sdn_tests)

    # Define simulation parameters
    Fs = exp_config.Fs

    duration = exp_config.duration

    # --- Source and Receiver Setup ---
    # Common receiver setup for both single and multi-source runs
    if use_grid:
        if GRID_SELECTION == "full":
             # Full grid covering the whole room
             receiver_positions = generate_full_receiver_grid(
                 active_room['width'], 
                 active_room['depth'], 
                 height=active_room['mic z'], 
                 n_x=4, 
                 n_y=4, 
                 margin=0.5
             )
             grid_tag = "FULLGRID" 
             print(f"Running in MULTI-POSITION (FULL-GRID) mode for room '{active_room['display_name']}'")
        
        elif GRID_SELECTION == "quarter":
             # Original corner/quadrant grid
             # Correctly use half the room dimensions for the grid, as in spatial_analysis.py
             receiver_positions = generate_receiver_grid_old(active_room['width'] / 2, active_room['depth'] / 2, margin=0.5, n_points=16)
             grid_tag = "room" # Matches legacy 'aes_room_spatial...'
             print(f"Running in MULTI-POSITION (QUARTER-GRID) mode for room '{active_room['display_name']}'")
        
        else:
            raise ValueError(f"Unknown GRID_SELECTION: {GRID_SELECTION}")
             
    else:
        receiver_positions = [(active_room['mic x'], active_room['mic y'], active_room['mic z'])]
        grid_tag = "single"
        print(f"\nRunning in SINGLE-POSITION mode for room '{active_room['display_name']}'")

    if PROCESS_MULTIPLE_SOURCES:
        print("\n--- PROCESSING MULTIPLE SOURCES ---")
        
        # Generate all possible sources
        all_sources = generate_source_positions(active_room)
        
        # Filter based on FILES_TO_PROCESS
        source_list = []
        for src_x, src_y, src_z, src_name in all_sources:
            # Build expected filename for this source
            room_name = active_room.get('display_name', 'unknown_room')
            
            # Construct filename base
            # If using full grid: "aes_FULLGRID"
            # If using old grid: "aes_quarter" (legacy quarterbehavior)
            if use_grid and grid_tag != "quarter":
                 room_prefix = room_name.split()[0].lower() # "aes"
                 filename_base = f"{room_prefix}_{grid_tag}"
            else:
                 filename_base = room_name.lower().replace(' ', '_')

            expected_filename = f"{filename_base}_{src_name.lower()}.npz"
            
            # Only include if in FILES_TO_PROCESS
            if expected_filename in FILES_TO_PROCESS:
                source_list.append((src_x, src_y, src_z, src_name))
                print(f"  Will process: {src_name} -> {expected_filename}")
            else:
                print(f"  Skipping: {src_name} (not in FILES_TO_PROCESS)")
        
        if not source_list:
            print("WARNING: No sources to process after filtering. Check FILES_TO_PROCESS.")
        
        for src_x, src_y, src_z, src_name in source_list:
            source_position = (src_x, src_y, src_z)
            print(f"\nProcessing for source: '{src_name}' at {source_position}")

            # Define a unique output path for this specific source
            # Use absolute path relative to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            output_dir = os.path.join(project_root, "results", "paper_data")
            
            room_name = active_room.get('display_name', 'unknown_room')
            
            # Reconstruct filename (logic must match above)
            if use_grid and grid_tag != "quarter":
                 room_prefix = room_name.split()[0].lower()
                 filename_base = f"{room_prefix}_{grid_tag}"
            else:
                 filename_base = room_name.lower().replace(' ', '_')
                 
            output_filename = f"{filename_base}_{src_name.lower()}.npz"
            output_path = os.path.join(output_dir, output_filename)

            # Run the calculation and saving process for the current source
            calculate_and_save_data(
                room_params=active_room,
                source_pos=source_position,
                receiver_positions=receiver_positions,
                duration=duration,
                Fs=Fs,
                method_configs=method_configs,
                output_path=output_path,
                update_mode=UPDATE_EXISTING_FILE
            )
    else:
        print("\n--- PROCESSING SINGLE SOURCE from room config ---")
        # Original behavior: use the single source from the config
        source_position = (active_room['source x'], active_room['source y'], active_room['source z'])

        # Try to identify source name from known sources
        src_name = "custom_source"
        all_possible = generate_source_positions(active_room)
        for sx, sy, sz, sname in all_possible:
             if np.allclose([sx, sy, sz], source_position, atol=1e-3):
                 src_name = sname
                 print(f"  Identified source as: {src_name}")
                 break
        
        if src_name == "custom_source":
             print("  Source does not match any known named position. Using 'custom_source'.")

        # Define output path
        # Use absolute path relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        output_dir = os.path.join(project_root, "results", "paper_data")
        
        room_name = active_room.get('display_name')
        
        # Reconstruct filename logic for single source too
        if use_grid and grid_tag != "quarter":
             room_prefix = room_name.split()[0].lower()
             filename_base = f"{room_prefix}_{grid_tag}"
        else:
             filename_base = room_name.lower().replace(' ', '_')
             
        output_filename = f"{filename_base}_{src_name.lower()}.npz"
        output_path = os.path.join(output_dir, output_filename)

        # Run the calculation and saving process
        calculate_and_save_data(
            room_params=active_room,
            source_pos=source_position,
            receiver_positions=receiver_positions,
            duration=duration,
            Fs=Fs,
            method_configs=method_configs,
            output_path=output_path,
            update_mode=UPDATE_EXISTING_FILE
        )


