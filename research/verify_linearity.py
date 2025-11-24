import numpy as np
import os
import matplotlib.pyplot as plt
from rir_calculators import calculate_sdn_rir
from geometry import Room, Source
from copy import deepcopy

# Setup paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
DATA_DIR = os.path.join(_project_root, "results", "paper_data")

# Test configuration
TEST_FILE = "aes_room_spatial_edc_data_center_source.npz"
FS = 44100
DURATION = 0.2 # Short duration for quick check

def verify_linearity():
    print("--- Verifying SDN Linearity for c-vector ---")
    
    # 1. Load a real room config
    data_path = os.path.join(DATA_DIR, TEST_FILE)
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    with np.load(data_path, allow_pickle=True) as data:
        room_params = data['room_params'][0]
        source_pos = data['source_pos']
        # Use first receiver for test
        mic_pos = data['receiver_positions'][0] 
        
    # Setup Room
    room = Room(room_params['width'], room_params['depth'], room_params['height'])
    num_samples = int(FS * DURATION)
    impulse = Source.generate_signal('dirac', num_samples)
    room.set_source(*source_pos, signal=impulse['signal'], Fs=FS)
    room.set_microphone(mic_pos[0], mic_pos[1], room_params['mic z'])
    
    room_params['reflection'] = np.sqrt(1 - room_params['absorption'])
    room.wallAttenuation = [room_params['reflection']] * 6
    
    # Base Config
    base_cfg = {
        'enabled': True,
        'label': "linearity_test",
        'flags': {
            'specular_source_injection': True,
            # We are testing the 'injection_c_vector' behavior
        }
    }

    # ---------------------------------------------------------
    # 1. Generate Ground Truth (Random c vector)
    # ---------------------------------------------------------
    # np.random.seed(42)
    # target_c = np.random.uniform(1.0, 7.0, size=6)
    target_c = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 1.5])
    print(f"Target c-vector: {target_c}")
    
    cfg_target = deepcopy(base_cfg)
    cfg_target['flags']['injection_c_vector'] = target_c.tolist()
    
    print("Computing Ground Truth RIR...")
    _, rir_true, _, _ = calculate_sdn_rir(room_params, "Truth", room, DURATION, FS, cfg_target)
    
    # ---------------------------------------------------------
    # 2. Reconstruct using Basis Functions
    # Formula: R(c) = R(0) + sum( c_i * (R(e_i) - R(0)) )
    # ---------------------------------------------------------
    print("Computing Basis RIRs...")
    
    # A) Baseline (All zeros)
    cfg_base = deepcopy(base_cfg)
    cfg_base['flags']['injection_c_vector'] = [1.] * 6
    _, rir_base, _, _ = calculate_sdn_rir(room_params, "Base", room, DURATION, FS, cfg_base)
    
    # B) Basis Deltas
    rir_recon = rir_base.copy()
    
    for i in range(6):
        # Create basis vector e_i (one wall = 1.0, others = 0.0)
        c_basis = [0.0] * 6
        c_basis[i] = 1.0
        
        cfg_basis = deepcopy(base_cfg)
        cfg_basis['flags']['injection_c_vector'] = c_basis
        
        _, rir_i, _, _ = calculate_sdn_rir(room_params, f"Basis_{i}", room, DURATION, FS, cfg_basis)
        
        # Calculate Slope/Delta for this wall
        # Delta = R(1) - R(0)
        delta_rir = rir_i - rir_base
        
        # Add weighted contribution
        # contribution = target_c[i] * delta_rir
        rir_recon += target_c[i] * delta_rir
        
    # ---------------------------------------------------------
    # 3. Compare
    # ---------------------------------------------------------
    # Crop to shortest length if necessary
    min_len = min(len(rir_true), len(rir_recon))
    rir_true = rir_true[:min_len]
    rir_recon = rir_recon[:min_len]

    error = rir_true - rir_recon
    max_error = np.max(np.abs(error))
    max_amp = np.max(np.abs(rir_true))
    rel_error = max_error / max_amp if max_amp > 0 else 0
    
    print(f"\nMax Absolute Error: {max_error:.10e}")
    print(f"Max Amplitude:      {max_amp:.10e}")
    print(f"Relative Error:     {rel_error:.10e}")
    
    if rel_error < 1e-9:
        print("\nSUCCESS: Linearity confirmed! The fast optimization method is valid.")
    else:
        print("\nWARNING: Large error detected. Linearity assumption might be flawed.")
        
    # Check injection mapping assumption
    # Just to confirm if c[0] maps to closest wall
    # We can inspect rir_basis_0 vs rir_basis_1 start times if needed, but not critical for linearity proof.

if __name__ == "__main__":
    verify_linearity()

