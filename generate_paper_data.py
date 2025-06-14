import numpy as np
import os
import json
from typing import Dict, List, Tuple
import geometry
import analysis as an
import EchoDensity as ned
from rir_calculators import calculate_pra_rir, calculate_sdn_rir, calculate_ho_sdn_rir, calculate_rimpy_rir
import pyroomacoustics as pra
from spatial_analysis import generate_receiver_grid_old

def calculate_and_save_data(room_params: dict, source_pos: Tuple[float, float, float],
                              receiver_positions: List[Tuple[float, float]],
                              duration: float, Fs: int,
                              method_configs: Dict,
                              output_path: str,
                              update_mode: bool = False):
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
        if method not in all_rirs:
            print(f"  RIRs not found for '{method}'. Calculating...")
            any_new_data_calculated = True
            current_method_rirs = []
            for i, (rx, ry) in enumerate(receiver_positions):
                print(f"    ... receiver {i+1}/{len(receiver_positions)} at ({rx:.2f}, {ry:.2f})")
                
                current_params_for_calc = room_parameters.copy()
                current_params_for_calc.update({
                    'mic x': rx,
                    'mic y': ry,
                    'source x': source_pos[0],
                    'source y': source_pos[1],
                    'source z': source_pos[2],
                })

                if method == 'ISM':
                    max_order = config.get('max_order', 100)
                    rir, _ = calculate_pra_rir(current_params_for_calc, duration, Fs, max_order)
                
                elif method.startswith('SDN-'):
                    room.set_microphone(rx, ry, room_params['mic z'])
                    _, rir, _, _ = calculate_sdn_rir(room_parameters, method, room, duration, Fs, config)

                elif method.startswith('HO-SDN'):
                    source_signal = config.get('source_signal', 'dirac')
                    order = config.get('order', 2)
                    rir, _ = calculate_ho_sdn_rir(current_params_for_calc, Fs, duration, source_signal, order=order)

                elif method.startswith('RIMPY'):
                    reflection_sign = config.get('reflection_sign')
                    rir, _ = calculate_rimpy_rir(current_params_for_calc, duration, Fs, reflection_sign=reflection_sign)

                else:
                    print(f"    Warning: Unknown or unsupported method {method}, skipping.")
                    continue
                
                if np.max(np.abs(rir)) > 0:
                    rir = rir / np.max(np.abs(rir))
                    
                current_method_rirs.append(rir)
            
            all_rirs[method] = np.array(current_method_rirs)
        else:
            print(f"  RIRs for '{method}' already exist. Skipping calculation.")

        # --- Step 2: Check and Calculate EDCs ---
        if method not in all_edcs and method in all_rirs:
            print(f"  EDCs not found for '{method}'. Calculating...")
            any_new_data_calculated = True
            edcs_for_method = []
            for rir in all_rirs[method]:
                edc, _, _ = an.compute_edc(rir, Fs, plot=False)
                edcs_for_method.append(edc)
            all_edcs[method] = np.array(edcs_for_method)
        elif method in all_edcs:
            print(f"  EDCs for '{method}' already exist. Skipping calculation.")

        # --- Step 3: Check and Calculate NEDs ---
        if method not in all_neds and method in all_rirs:
            print(f"  NEDs not found for '{method}'. Calculating...")
            any_new_data_calculated = True
            neds_for_method = []
            for rir in all_rirs[method]:
                ned_profile = ned.echoDensityProfile(rir, fs=Fs)
                neds_for_method.append(ned_profile)
            all_neds[method] = np.array(neds_for_method)
        elif method in all_neds:
            print(f"  NEDs for '{method}' already exist. Skipping calculation.")


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
            
        np.savez(output_path, **save_dict)
        print("--- Save complete ---")
    else:
        print("\n--- No new data was calculated. File is already up-to-date. ---")


if __name__ == "__main__":
    # --- EXPERIMENT SETUP ---
    # Set to True to load the existing data file and only run/replace
    # the methods that are marked 'enabled' in this script.
    UPDATE_EXISTING_FILE = True
    ENABLED = True

    # --- Room Setups ---
    room_waspaa = {
        'display_name': 'WASPAA Room',
        'width': 6, 'depth': 7, 'height': 4,
        'source x': 3.6, 'source y': 5.3, 'source z': 1.3,
        'mic x': 1.2, 'mic y': 1.8, 'mic z': 2.4,
        'absorption': 0.1,
    }
    room_aes = {
        'display_name': 'AES Room',
        'width': 9, 'depth': 7, 'height': 4,
        'source x': 4.5, 'source y': 3.5, 'source z': 2,
        'mic x': 2, 'mic y': 2, 'mic z': 1.5,
        'absorption': 0.2,
    }
    room_journal = {
        'display_name': 'Journal Room',
        'width': 3.2, 'depth': 4, 'height': 2.7,
        'source x': 2, 'source y': 3., 'source z': 2,
        'mic x': 1.6, 'mic y': 2, 'mic z': 1.8,
        'absorption': 0.1,
    }

    # room_journal = {
    #     'display_name': 'Journal Room',
    #     'width': 3.2, 'depth': 4, 'height': 2.7,
    #     'source x': 1, 'source y': 1.9, 'source z': 1.6,
    #     'mic x': 2, 'mic y': 1, 'mic z': 1.5,
    #     'absorption': 0.1,
    # }

    # Choose which room configuration to use for this run
    active_room = room_waspaa

    # Define the methods to be calculated, based on main.py
    method_configs = {
        'ISM': {
            'enabled': ENABLED,
            'info': 'pra 100',
            'label': "ISM",
            'max_order': 100
        },


        'SDN-Test1': { # Renamed from Test0 to match paper_figures.py
            'enabled': ENABLED,
            'info': "c1 original",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 1,
            },
            'label': "SDN"
        },

        'SDN-Test_3': {
            'enabled': ENABLED,
            'info': "c -3",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': -3,
            },
            'label': "SDN"
        },

        'SDN-Test_2': {
            'enabled': ENABLED,
            'info': "c -2",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': -2,
            },
            'label': "SDN"
        },

        'SDN-Test3': {
            'enabled': ENABLED,
            'info': "c3",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 3,
            },
            'label': "SDN"
        },
        'SDN-Test4': {
            'enabled': ENABLED,
            'info': "c4",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 4,
            },
            'label': "SDN"
        },
        'SDN-Test5': {
            'enabled': ENABLED,
            'info': "c5",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 5,
            },
            'label': "SDN Test 5"
        },
        'SDN-Test6': {
            'enabled': ENABLED,
            'info': "c6",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 6,
            },
            'label': "SDN Test 6"
        },
        'SDN-Test7': {
            'enabled': ENABLED,
            'info': "c7",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 7,
            },
            'label': "SDN Test 7"
        },
        'HO-SDN-N2': {
            'enabled': ENABLED,
            'info': 'HO-SDN N=2',
            'source_signal': 'dirac',
            'order': 2,
            'label': 'HO-SDN N2'
        },
        'HO-SDN-N3': {
            'enabled': ENABLED,
            'info': 'HO-SDN N=3',
            'source_signal': 'dirac',
            'order': 3,
            'label': 'HO-SDN N3'
        },
    
        'RIMPY-neg': {
            'enabled': ENABLED,
            'info': 'Negative Reflection',
            'reflection_sign': -1,
            'label': 'RIMPY'
        },
        'Test_trial': {
            'enabled': False,
            'info': "testing",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': -20,
            },
            'label': "SDN"
        },
    }

    # Define simulation parameters
    Fs = 44100

    if active_room['display_name'] == 'Journal Room':
        duration = 1.2
    elif active_room['display_name'] == 'WASPAA Room':
        duration = 1.8
    elif active_room['display_name'] == 'AES Room':
        duration = 1

    # --- Source and Receiver Setup ---
    source_position = (active_room['source x'], active_room['source y'], active_room['source z'])
    
    # By default, run for the single mic position defined in the active_room.
    receiver_positions = [(active_room['mic x'], active_room['mic y'])]
    print(f"\nRunning in SINGLE-POSITION mode for room '{active_room['display_name']}'")

    # To run for a grid of receivers, uncomment the line below.
    # receiver_positions = generate_receiver_grid_old(active_room['width'], active_room['depth'], margin=1.0, n_points=16)
    # print(f"Running in MULTI-POSITION (GRID) mode for room '{active_room['display_name']}'")


    # Define output path
    output_dir = "results/paper_data"
    room_name = active_room.get('display_name', 'unknown_room')
    filename_suffix = room_name.lower().replace(' ', '_')
    output_filename = f"{filename_suffix}_spatial_edc_data.npz"
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