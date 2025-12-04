"""
FINAL DEFINITIVE TEST:
Run optimisation_singleC.py EXACT logic with fresh cache
and compare with paper_figures result
"""
import numpy as np
import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import geometry
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation, _BASIS_CACHE
from analysis import analysis as an

# CLEAR CACHE
print("Clearing basis cache...")
_BASIS_CACHE.clear()

DATA_FILE = 'results/paper_data/aes_room_spatial_edc_data_top_middle_source.npz'

# Load data
with np.load(DATA_FILE, allow_pickle=True) as data:
    room_parameters = data['room_params'][0]
    source_pos = data['source_pos']
    receiver_positions = data['receiver_positions']
    Fs = int(data['Fs'])
    duration = float(data['duration'])
    ref_edcs = data['edcs_RIMPY-neg10']
    saved_sdn_edcs = data['edcs_SDN-Test2.998']

# Setup room (EXACT optimisation_singleC.py logic from lines 239-245)
room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

# Base config (lines 247-250)
base_sdn_config = {
    'enabled': True,
    'info': "c_optimized",
    'flags': {'specular_source_injection': True}
}

# Compute RMSE (EXACT compute_spatial_rmse logic from lines 80-123)
c_val = 2.998
cfg = deepcopy(base_sdn_config)
cfg['flags']['source_weighting'] = c_val
cfg['use_fast_method'] = True
cfg['label'] = f'SDN-SW-c_{c_val:.2f}'

print(f"\n=== COMPUTING WITH FRESH CACHE, c={c_val} ===")
print(f"Cache contains {len(_BASIS_CACHE)} entries")

rmse_fresh_list = []
for i, (rx, ry) in enumerate(receiver_positions):
    room.set_microphone(rx, ry, room_parameters['mic z'])
    cfg['cache_label'] = f"top_middle_rx{i:02d}"
    
    _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_parameters, "SDN-Opt", room, duration, Fs, cfg)
    rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']
    edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)
    
    rmse = an.compute_RMS(
        edc_sdn, ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_fresh_list.append(rmse)
    
    if i == 0:
        print(f"First receiver computed. Cache now has {len(_BASIS_CACHE)} entries")

mean_rmse_fresh = np.mean(rmse_fresh_list)

# Compare with saved
rmse_saved_list = []
for i in range(len(receiver_positions)):
    rmse = an.compute_RMS(
        saved_sdn_edcs[i], ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_saved_list.append(rmse)

mean_rmse_saved = np.mean(rmse_saved_list)

print(f"\n=== RESULTS ===")
print(f"RMSE (fresh Fast with cleared cache): {mean_rmse_fresh:.6f}")
print(f"RMSE (saved data from file):         {mean_rmse_saved:.6f}")
print(f"Difference:                           {abs(mean_rmse_fresh - mean_rmse_saved):.6f}")
print(f"Percent difference:                   {abs(mean_rmse_fresh - mean_rmse_saved) / mean_rmse_saved * 100:.2f}%")

if abs(mean_rmse_fresh - mean_rmse_saved) < 0.001:
    print("\n✓ MATCH - No discrepancy when using fresh cache!")
else:
    print(f"\n✗ DISCREPANCY CONFIRMED - {abs(mean_rmse_fresh - mean_rmse_saved) / mean_rmse_saved * 100:.1f}% difference with fresh cache")
