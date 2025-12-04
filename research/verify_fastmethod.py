import numpy as np
import os
import matplotlib.pyplot as plt
from rir_calculators import calculate_sdn_rir, rir_normalisation
from geometry import Room, Source
from copy import deepcopy
from analysis import plot_room as pp
import geometry
# Setup paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
DATA_DIR = os.path.join(_project_root, "results", "paper_data")
TEST_FILE = "aes_room_spatial_edc_data_center_source.npz"

def verify_interpolation():
    print("--- Verifying Analytical RIR Reconstruction (Interpolation) ---")

    # 1. Setup Room
    # with np.load(os.path.join(DATA_DIR, TEST_FILE), allow_pickle=True) as data:
    #     room_params = data['room_params'][0]
    #     source_pos = data['source_pos']
    #     mic_pos = data['receiver_positions'][0]
    #
    Fs = 44100
    duration = 0.5
    num_samples = int(Fs * duration)
    #
    # room = Room(room_params['width'], room_params['depth'], room_params['height'])
    # room.set_microphone(mic_pos[0], mic_pos[1], room_params['mic z'])
    # impulse = Source.generate_signal('dirac', num_samples)
    # room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)

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
    
    # Base config
    base_cfg = {
        'enabled': True,
        'label': "Test",
        'flags': {'specular_source_injection': True}
    }

    # ---------------------------------------------------------
    # 2. Pre-compute Basis RIRs (The "Training" Phase)
    # ---------------------------------------------------------
    print("\n1. Pre-computing Basis Functions...")
    
    # Run with c=0 (The "Constant Part")
    print("   Running simulation for c=0...")
    cfg_0 = deepcopy(base_cfg)
    cfg_0['flags']['source_weighting'] = 0

    _, rir_base, _, _ = calculate_sdn_rir(room_parameters, "Rbase (c=0)", room, duration, Fs, cfg_0)
    
    # Run with c=1 (To find the "Variable Part" slope)
    print("   Running simulation for c=1...")
    cfg_1 = deepcopy(base_cfg)
    cfg_1['flags']['source_weighting'] = 1
    _, rir_c1, _, _ = calculate_sdn_rir(room_parameters, "c=1", room, duration, Fs, cfg_1)
    
    # Calculate the Slope RIR
    # Slope = R(1) - R(0)
    rir_shape= rir_c1 - rir_base
    
    # ---------------------------------------------------------
    # 3. Predict and Verify for a Random 'c'
    # ---------------------------------------------------------
    test_c = 5 # Arbitrary float value
    print(f"\n2. Testing Linear Prediction for c = {test_c}")
    
    # A. Analytic Prediction (Instant)
    # R(c) = R(0) + c * Slope
    print("   Calculating Analytic Prediction...")
    rir_fast = rir_base + (test_c * rir_shape)
    
    # B. Ground Truth Simulation (Slow)
    print(f"   Running Ground Truth Simulation for c={test_c}...")
    cfg_test = deepcopy(base_cfg)
    cfg_test['flags']['source_weighting'] = 5
    _, rir_truth_c5, _, _ = calculate_sdn_rir(room_parameters, f"c={test_c}", room, duration, Fs, cfg_test)
    
    # ---------------------------------------------------------
    # 4. Comparison
    # ---------------------------------------------------------
    error = rir_truth_c5 - rir_fast
    max_error = np.max(np.abs(error))
    max_amp = np.max(np.abs(rir_truth_c5))
    
    print("\nResults:")
    print(f"Max Absolute Error: {max_error:.10e}")
    print(f"Signal Peak:        {max_amp:.10e}")
    
    if max_error < 1e-9:
        print("\n✅ VERIFIED: Analytic reconstruction matches simulation perfectly.")
        print("   This means we can optimize 'c' without running the simulation loop!")
    else:
        print("\n❌ FAILED: Prediction does not match simulation.")

    # Plot
    rirs_plot = {
        "Truth": rir_truth_c5,
        "Predicted": rir_fast,
        "Error (x1e10)": error * 1e10
    }
    pp.create_interactive_rir_plot(rirs_plot, Fs)
    plt.show()

if __name__ == "__main__":
    verify_interpolation()

