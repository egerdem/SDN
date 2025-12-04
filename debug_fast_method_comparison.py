"""
Compare RMSE values when using c=2.998 exactly as stored in the file
vs freshly calculated with the Fast method
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import geometry
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation
from analysis import analysis as an

# Configuration
DATA_FILE = 'results/paper_data/aes_room_spatial_edc_data_top_middle_source.npz'
C_VALUE = 2.998  # EXACT value from the saved data
REFERENCE_METHOD = 'RIMPY-neg10'

print("="*80)
print(f"RMSE Comparison: Saved (c=2.998) vs Fresh Fast Method (c=2.998)")
print("="*80)

# Load data
with np.load(DATA_FILE, allow_pickle=True) as data:
    room_params = data['room_params'][0]
    source_pos = data['source_pos']
    receiver_positions = data['receiver_positions']
    Fs = int(data['Fs'])
    duration = float(data['duration'])
    
    # Reference EDCs (same for both)
    edc_ref = data[f'edcs_{REFERENCE_METHOD}']
    
    # Saved SDN EDCs
    edc_saved = data['edcs_SDN-Test2.998']

print(f"\nLoaded data:")
print(f"  Source: {source_pos}")
print(f"  Receivers: {len(receiver_positions)}")
print(f"  Ref EDC shape: {edc_ref.shape}")
print(f"  Saved SDN EDC shape: {edc_saved.shape}")

# Setup room for fresh calculation using FAST method
print(f"\n1. Calculating FRESH SDN with Fast method, c={C_VALUE}")
room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
room_params['reflection'] = np.sqrt(1 - room_params['absorption'])
room.wallAttenuation = [room_params['reflection']] * 6

sdn_config = {
    'enabled': True,
    'flags': {
        'specular_source_injection': True,
        'source_weighting': C_VALUE,
    },
    'use_fast_method': True,
    'label': "SDN-Fresh-Fast"
}

# Calculate fresh using FAST method (like optimisation_singleC.py does)
edc_fresh_fast = []
for i, (rx, ry) in enumerate(receiver_positions):
    room.set_microphone(rx, ry, room_params['mic z'])
    
    # Add cache label
    sdn_config['cache_label'] = f"top_middle_rx{i:02d}"
    
    # Use FAST method
    _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_params, "SDN-Opt", room, duration, Fs, sdn_config)
    rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']
    edc, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)
    edc_fresh_fast.append(edc)

edc_fresh_fast = np.array(edc_fresh_fast)
print(f"  Fresh Fast EDC shape: {edc_fresh_fast.shape}")

# Calculate RMSEs
print(f"\n2. Computing RMSE for each receiver")
print(f"-" * 80)
print(f"{'Receiver':<12} {'Position':<15} {'Saved RMSE':<13} {'Fresh RMSE':<13} {'Difference'}")
print(f"-" * 80)

rmse_saved_list = []
rmse_fresh_list = []

for i in range(len(receiver_positions)):
    rx, ry = receiver_positions[i]
    
    # RMSE with saved EDC
    rmse_saved = an.compute_RMS(
        edc_saved[i], edc_ref[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    
    # RMSE with fresh Fast EDC
    rmse_fresh = an.compute_RMS(
        edc_fresh_fast[i], edc_ref[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    
    rmse_saved_list.append(rmse_saved)
    rmse_fresh_list.append(rmse_fresh)
    
    diff = rmse_fresh - rmse_saved
    percent = (diff / rmse_saved * 100) if rmse_saved > 0 else 0
    
    print(f"Receiver {i+1:2d}  ({rx:.2f},{ry:.2f})     {rmse_saved:.6f}      {rmse_fresh:.6f}      {diff:+.6f} ({percent:+.1f}%)")

mean_saved = np.mean(rmse_saved_list)
mean_fresh = np.mean(rmse_fresh_list)
mean_diff = mean_fresh - mean_saved
mean_percent = (mean_diff / mean_saved * 100) if mean_saved > 0 else 0

print(f"-" * 80)
print(f"{'Mean':<12} {'':<15} {mean_saved:.6f}      {mean_fresh:.6f}      {mean_diff:+.6f} ({mean_percent:+.1f}%)")
print(f"=" * 80)

print(f"\n3. CONCLUSION")
print(f"-" * 80)
if abs(mean_diff) < 0.001:
    print("✓ SAVED and FRESH (Fast method) produce IDENTICAL results!")
    print("  The discrepancy must be due to different c values being compared.")
else:
    print(f"✗ Significant difference found: {mean_percent:+.1f}%")
    print(f"  paper_figures_spatial.py (saved):     {mean_saved:.6f}")
    print(f"  optimisation_singleC.py (fresh fast): {mean_fresh:.6f}")
    print(f"\n  This proves the saved data uses different calculation logic!")
print(f"=" * 80)
