"""
Reproduce EXACT flow ofgenerate_paper_data.py for SDN-Test2.998
Then compare with optimization flow
"""
import numpy as np
import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import geometry
import experiment_configs as exp_config
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation
from analysis import analysis as an

print("=" * 90)
print("REPRODUCING EXACT generate_paper_data.py FLOW")
print("=" * 90)

# Use the SAME config as generate_paper_data.py
sdn_tests = exp_config.sdn_tests
method = 'SDN-Test2.998'
config = sdn_tests[method]

print(f"\nMethod: {method}")
print(f"Config from experiment_configs.py:")
print(f"  {config}")
print(f"  Has 'use_fast_method': {'use_fast_method' in config}")

# Load room and source data
DATA_FILE = 'results/paper_data/aes_room_spatial_edc_data_top_middle_source.npz'
with np.load(DATA_FILE, allow_pickle=True) as data:
    room_params = data['room_params'][0]
    source_pos = data['source_pos']
    receiver_positions = data['receiver_positions']
    Fs = int(data['Fs'])
    duration = float(data['duration'])
    ref_edcs = data['edcs_RIMPY-neg10']

# Setup room (as generate_paper_data.py does)
room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
room.wallAttenuation = [np.sqrt(1 - room_params['absorption'])] * 6
room_parameters = room_params.copy()
room_parameters['reflection'] = np.sqrt(1 - room_params['absorption'])

# Calculate RIRs and EDCs EXACTLY as generate_paper_data.py does (lines 105-130, 142-152)

print(f"\n=== FLOW 1: generate_paper_data.py logic ===")
print(f"Calculating RIRs...")

edcs_generated = []
for i, (rx, ry) in enumerate(receiver_positions):
    room.set_microphone(rx, ry, room_params['mic z'])
    
    current_params_for_calc = room_parameters.copy()
    current_params_for_calc.update({
        'mic x': rx,
        'mic y': ry,
        'source x': source_pos[0],
        'source y': source_pos[1],
        'source z': source_pos[2],
    })
    
    # EXACT call as line 113 in generate_paper_data.py
    _, rir, _, _ = calculate_sdn_rir_fast(current_params_for_calc, method, room, duration, Fs, config)
    
    # EXACT normalization as lines 122-126
    if np.max(np.abs(rir)) > 0:
        normalized_rir_dict = rir_normalisation(rir, room, Fs, normalize_to_first_impulse=True)
        rir = next(iter(normalized_rir_dict.values()))
    
    # Compute EDC (as line 145)
    edc, _, _ = an.compute_edc(rir, Fs, plot=False)
    edcs_generated.append(edc)
    
    if i == 0:
        print(f"  First receiver: RIR len={len(rir)}, EDC len={len(edc)}")

edcs_generated = np.array(edcs_generated)
print(f"Generated EDCs shape: {edcs_generated.shape}")

# Compute RMSE using generated EDCs
print(f"\nComputing RMSE...")
rmse_generated = []
for i in range(len(receiver_positions)):
    rmse = an.compute_RMS(
        edcs_generated[i], ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_generated.append(rmse)

mean_rmse_generated = np.mean(rmse_generated)
print(f"Mean RMSE (generated flow): {mean_rmse_generated:.6f}")

# Now compare with saved data
with np.load(DATA_FILE, allow_pickle=True) as data:
    edcs_saved = data['edcs_SDN-Test2.998']

rmse_saved = []
for i in range(len(receiver_positions)):
    rmse = an.compute_RMS(
        edcs_saved[i], ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_saved.append(rmse)

mean_rmse_saved = np.mean(rmse_saved)
print(f"Mean RMSE (saved data):     {mean_rmse_saved:.6f}")

# Now try optimization flow
print(f"\n=== FLOW 2: optimisation_singleC.py logic ===")

sdn_config = {
    'enabled': True,
    'flags': {'specular_source_injection': True, 'source_weighting': 2.998},
    'use_fast_method': True,  # <-- Optimization explicitly sets this
    'label': 'SDN-SW-c_2.99'
}

edcs_optimization = []
for i, (rx, ry) in enumerate(receiver_positions):
    room.set_microphone(rx, ry, room_params['mic z'])
    
    sdn_config['cache_label'] = f"top_middle_rx{i:02d}"
    
    _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_parameters, "SDN-Opt", room, duration, Fs, sdn_config)
    rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']
    edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)
    edcs_optimization.append(edc_sdn)

edcs_optimization = np.array(edcs_optimization)

rmse_optimization = []
for i in range(len(receiver_positions)):
    rmse = an.compute_RMS(
        edcs_optimization[i], ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_optimization.append(rmse)

mean_rmse_optimization = np.mean(rmse_optimization)
print(f"Mean RMSE (optimization flow): {mean_rmse_optimization:.6f}")

print(f"\n=== COMPARISON ===")
print(f"Generated (like paper data): {mean_rmse_generated:.6f}")
print(f"Saved (actual paper data):   {mean_rmse_saved:.6f}")
print(f"Optimization flow:           {mean_rmse_optimization:.6f}")

print(f"\n=== CONCLUSION ===")
if abs(mean_rmse_generated - mean_rmse_optimization) < 0.001:
    print("✓ Both flows produce IDENTICAL results")
    print("  The saved data must be outdated or generated differently!")
else:
    print(f"✗ Different flows produce different results:")
    print(f"  Difference: {abs(mean_rmse_generated - mean_rmse_optimization):.6f}")
    print(f"  This suggests different config handling in calculate_sdn_rir_fast")
