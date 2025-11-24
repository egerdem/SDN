import numpy as np
import os
import matplotlib.pyplot as plt
from rir_calculators import calculate_sdn_rir, rir_normalisation
from geometry import Room, Source
import geometry
from copy import deepcopy
from analysis import plot_room as pp # Import plotting tools

# Setup paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
DATA_DIR = os.path.join(_project_root, "results", "paper_data")

# Test configuration
TEST_FILE = "aes_room_spatial_edc_data_center_source.npz"
FS = 44100
DURATION = 1 # Slightly longer for visual inspection

def verify_superposition():
    print("--- Verifying SDN Superposition (Subtraction Method) ---")

    room_aes = {
        'display_name': 'AES Room',
        'width': 9, 'depth': 7, 'height': 4,
        'source x': 4.5, 'source y': 3.5, 'source z': 2,
        'mic x': 2, 'mic y': 2, 'mic z': 1.5,
        'absorption': 0.2,
    }
    room_parameters = room_aes

    # print room name and duration of the experiment
    print(f"\n=== {room_parameters['display_name']} ===")
    Fs = 44100
    duration = 1  # seconds
    num_samples = int(Fs * duration)
    impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)

    print(f"Duration: {duration} seconds")

    room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
    room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
    room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                    signal="will be replaced", Fs=Fs)

    room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])

    # Setup signal
    room.source.signal = impulse_dirac['signal']
    # room.source.signal = impulse_gaussian['signal']

    # Calculate reflection coefficient - will be overwritten for each method
    room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
    room.wallAttenuation = [room_parameters['reflection']] * 6
    
    # 2. Reference Run: All walls active (Standard SDN)
    target_c = [5.0] * 6 # Standard SDN case
    
    base_cfg = {
        'enabled': True,
        'label': "Full SDN",
        'flags': {
            'specular_source_injection': True,
            'injection_c_vector': target_c
        }
    }
    
    print("Running Standard SDN (All walls active)...")
    _, rir_full, _, _ = calculate_sdn_rir(room_parameters, "Full", room, DURATION, FS, base_cfg)
    
    # 3. Component Runs
    print("Running Component Simulations...")
    
    # A. Direct Sound Only (Baseline)
    cfg_direct = deepcopy(base_cfg)
    cfg_direct['flags']['source_pressure_injection_coeff'] = [0.0] * 6
    
    print("  Calculating Baseline (Direct Sound Only)...")
    _, rir_baseline, _, _ = calculate_sdn_rir(room_parameters, "Baseline", room, DURATION, FS, cfg_direct)
    
    # B. Walls
    rir_recon = rir_baseline.copy()
    
    num_walls = 6
    for i in range(num_walls):
        coeffs = [0.0] * num_walls
        coeffs[i] = 0.5 
        
        cfg_single = deepcopy(base_cfg)
        cfg_single['flags']['source_pressure_injection_coeff'] = coeffs
        
        _, rir_single, _, _ = calculate_sdn_rir(room_parameters, f"Wall_{i}", room, DURATION, FS, cfg_single)
        
        reverb_part = rir_single - rir_baseline
        rir_recon += reverb_part
        
    # 4. Compare
    error = rir_full - rir_recon
    max_error = np.max(np.abs(error))
    max_amp = np.max(np.abs(rir_full))
    
    print(f"\nMax Absolute Error: {max_error:.10e}")
    print(f"Max Amplitude:      {max_amp:.10e}")
    
    if max_error < 1e-9:
        print("\nSUCCESS: Superposition holds.")
    else:
        print("\nWARNING: Superposition failed.")

    # 5. Visual Comparison
    print("\nLaunching Interactive Plot...")
    
    rirs_to_plot = {
        "Reference (Full)": rir_full,
        "Reconstructed": rir_recon,
        "Baseline (Direct Only)": rir_baseline,
        "Error (x1e10)": error * 1e10 
    }
    
    # Normalize for display
    rirs_norm = rir_normalisation(rirs_to_plot, room, FS, normalize_to_first_impulse=True)

    pp.create_unified_interactive_plot(rirs_norm, Fs, room_parameters)
    plt.show()

if __name__ == "__main__":
    verify_superposition()
