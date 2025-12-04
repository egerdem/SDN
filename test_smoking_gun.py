"""
Direct comparison: Force optimisation_singleC logic to evaluate at c=2.998
and compare with what paper_figures_spatial.py gets from saved data
"""
import numpy as np
import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import geometry
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation
from analysis import analysis as an

# Load the EXACT same data file that both scripts use
DATA_FILE = 'results/paper_data/aes_room_spatial_edc_data_top_middle_source.npz'
C_VALUE = 2.998  # Exact value from saved data
REFERENCE_METHOD = 'RIMPY-neg10'

print("="*90)
print("CRITICAL TEST: Same c value, same reference EDCs, same everything")
print("="*90)

# Load data (as optimisation_singleC.py does on line 215-222)
with np.load(DATA_FILE, allow_pickle=True) as data:
    room_parameters = data['room_params'][0]
    source_pos = data['source_pos']
    receiver_positions = data['receiver_positions']
    Fs = int(data['Fs'])
    duration = float(data['duration'])
    
    # Load reference EDCs (THIS IS KEY - same as both scripts use)
    ref_edcs = data[f'edcs_{REFERENCE_METHOD}']
    
    # For comparison: also load saved SDN EDCs
    saved_sdn_edcs = data['edcs_SDN-Test2.998']

print(f"\nLoaded from file:")
print(f"  Source position: {source_pos}")
print(f"  Number of receivers: {len(receiver_positions)}")
print(f"  Reference EDC shape: {ref_edcs.shape}")
print(f"  Saved SDN EDC shape: {saved_sdn_edcs.shape}")

# ==================================================================
# PART 1: Calculate RMSE using SAVED SDN EDCs (paper_figures method)
# ==================================================================
print(f"\n{'='*90}")
print("PART 1: RMSE using SAVED SDN EDCs (paper_figures_spatial.py method)")
print(f"{'='*90}")

rmse_from_saved = []
for i in range(len(receiver_positions)):
    rmse = an.compute_RMS(
        saved_sdn_edcs[i], ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_from_saved.append(rmse)
    print(f"Receiver {i+1:2d}: {rmse:.6f}  (using saved SDN EDC)")

mean_rmse_saved = np.mean(rmse_from_saved)
print(f"\n>>> Mean RMSE (from SAVED EDCs): {mean_rmse_saved:.6f}")

# ==================================================================
# PART 2: Calculate RMSE using FRESH calculation (optimisation method)
# ==================================================================
print(f"\n{'='*90}")
print("PART 2: RMSE using FRESH calculation (optimisation_singleC.py method)")
print(f"{'='*90}")

# Setup room (lines 239-245 in optimisation_singleC.py)
room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

# Base SDN config (lines 247-250)
base_sdn_config = {
    'enabled': True,
    'info': f"c={C_VALUE}",
    'flags': {'specular_source_injection': True}
}

# Now calculate as optimisation_singleC.py does (lines 95-123)
cfg = deepcopy(base_sdn_config)
cfg['flags']['source_weighting'] = C_VALUE
cfg['use_fast_method'] = True
cfg['label'] = f'SDN-SW-c_{C_VALUE:.2f}'

rmse_from_fresh = []
for i, (rx, ry) in enumerate(receiver_positions):
    # Update mic position
    room.set_microphone(rx, ry, room_parameters['mic z'])
    
    # Add cache label
    cfg['cache_label'] = f"top_middle_rx{i:02d}"
    
    # Calculate RIR using Fast method
    _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_parameters, "SDN-Opt", room, duration, Fs, cfg)
    rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']
    
    # Compute EDC
    edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)
    
    # Get reference EDC for this receiver
    ref_edc = ref_edcs[i]
    
    print(f"Receiver {i+1:2d}: rir_len={len(rir_sdn_normed)}, edc_len={len(edc_sdn)}, ref_edc_len={len(ref_edc)}")
    
    # Compute RMSE (EXACT same call as optimisation_singleC.py line 118-120)
    rmse = an.compute_RMS(
        edc_sdn, ref_edc,
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_from_fresh.append(rmse)
    print(f"           RMSE = {rmse:.6f}")

mean_rmse_fresh = np.mean(rmse_from_fresh)
print(f"\n>>> Mean RMSE (from FRESH calculation): {mean_rmse_fresh:.6f}")

# ==================================================================
# PART 3: Direct comparison
# ==================================================================
print(f"\n{'='*90}")
print("PART 3: COMPARISON")
print(f"{'='*90}")

print(f"\n{'Receiver':<12} {'Saved EDC':<13} {'Fresh EDC':<13} {'Difference':<15} {'% Diff'}")
print("-" * 90)

for i in range(len(receiver_positions)):
    diff = rmse_from_fresh[i] - rmse_from_saved[i]
    pct = (diff / rmse_from_saved[i] * 100) if rmse_from_saved[i] > 0 else 0
    rx, ry = receiver_positions[i]
    print(f"{i+1:2d} ({rx:.2f},{ry:.2f})  {rmse_from_saved[i]:.6f}      {rmse_from_fresh[i]:.6f}      {diff:+.6f}          {pct:+.2f}%")

print("-" * 90)
diff_mean = mean_rmse_fresh - mean_rmse_saved
pct_mean = (diff_mean / mean_rmse_saved * 100) if mean_rmse_saved > 0 else 0
print(f"{'MEAN':<12} {mean_rmse_saved:.6f}      {mean_rmse_fresh:.6f}      {diff_mean:+.6f}          {pct_mean:+.2f}%")

print(f"\n{'='*90}")
print("CONCLUSION:")
print(f"{'='*90}")

if abs(pct_mean) < 0.1:
    print("✓ IDENTICAL! Both methods produce the same RMSE.")
    print("  The discrepancy you observed must be from comparing different c values.")
else:
    print(f"✗ DIFFERENT! {pct_mean:+.2f}% discrepancy found!")
    print(f"\n  This is the SMOKING GUN - same c value, different RMSEs:")
    print(f"    - Saved data (paper_figures):  {mean_rmse_saved:.6f}")
    print(f"    - Fresh calc (optimisation):   {mean_rmse_fresh:.6f}")
    print(f"\n  Possible causes to investigate:")
    print(f"    1. Reference EDC array lengths differ")
    print(f"    2. Different EDC computation parameters")
    print(f"    3. Different normalization in saved vs fresh")
    print(f"    4. Different room setup parameters")

print(f"{'='*90}")
