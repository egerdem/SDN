"""
Compare saved EDCs from file vs freshly calculated EDCs
to identify the source of RMSE discrepancy between paper_figures_spatial.py 
and optimisation_singleC.py
"""
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import geometry
from rir_calculators import calculate_sdn_rir, calculate_sdn_rir_fast, rir_normalisation
from analysis import analysis as an

# Configuration
DATA_FILE = 'results/paper_data/aes_room_spatial_edc_data_top_middle_source.npz'
METHOD = 'SDN-Test2.998'
C_VALUE = 2.998
REFERENCE_METHOD = 'RIMPY-neg10'
TEST_RECEIVER_INDEX = 0  # First receiver for detailed comparison

print("="*80)
print("EDC COMPARISON: Saved vs Fresh Calculation")
print("="*80)

# Load saved data
print(f"\n1. Loading saved data from: {DATA_FILE}")
with np.load(DATA_FILE, allow_pickle=True) as data:
    room_params = data['room_params'][0]
    source_pos = data['source_pos']
    receiver_positions = data['receiver_positions']
    Fs = int(data['Fs'])
    duration = float(data['duration'])
    
    # Saved EDCs
    edc_saved = data[f'edcs_{METHOD}']
    edc_ref = data[f'edcs_{REFERENCE_METHOD}']
    
    print(f"   Room: {room_params['width']}x{room_params['depth']}x{room_params['height']} m")
    print(f"   Source: {source_pos}")
    print(f"   Receivers: {len(receiver_positions)}")
    print(f"   Fs: {Fs} Hz, Duration: {duration} s")
    print(f"   Saved EDC shape: {edc_saved.shape}")

# Setup room for fresh calculation
print(f"\n2. Setting up room for fresh SDN calculation (c={C_VALUE})")
room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
room_params['reflection'] = np.sqrt(1 - room_params['absorption'])
room.wallAttenuation = [room_params['reflection']] * 6

# SDN configuration
sdn_config = {
    'enabled': True,
    'info': f"c={C_VALUE}",
    'flags': {
        'specular_source_injection': True,
        'source_weighting': C_VALUE,
    },
    'label': "SDN-Test"
}

# Calculate fresh EDCs for all receivers
print(f"\n3. Calculating fresh SDN RIRs and EDCs for all {len(receiver_positions)} receivers")
edc_fresh = []
for i, (rx, ry) in enumerate(receiver_positions):
    room.set_microphone(rx, ry, room_params['mic z'])
    
    # Use Standard method (same as generate_paper_data.py)
    _, rir_sdn, _, _ = calculate_sdn_rir(room_params, "SDN-Fresh", room, duration, Fs, sdn_config)
    
    # Normalize (same as generate_paper_data.py)
    rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']
    
    # Compute EDC
    edc, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)
    edc_fresh.append(edc)
    
    if i == 0:
        print(f"   First receiver RIR length: {len(rir_sdn)}, EDC length: {len(edc)}")

edc_fresh = np.array(edc_fresh)
print(f"   Fresh EDC shape: {edc_fresh.shape}")

# Pad to same length if needed
print(f"\n4. Ensuring EDC arrays have same length")
max_len = max(edc_saved.shape[1], edc_fresh.shape[1])
if edc_saved.shape[1] < max_len:
    print(f"   Padding saved EDCs from {edc_saved.shape[1]} to {max_len}")
    edc_saved_padded = np.array([np.pad(e, (0, max_len - len(e)), 'constant', constant_values=e[-1]) 
                                 for e in edc_saved])
else:
    edc_saved_padded = edc_saved

if edc_fresh.shape[1] < max_len:
    print(f"   Padding fresh EDCs from {edc_fresh.shape[1]} to {max_len}")
    edc_fresh_padded = np.array([np.pad(e, (0, max_len - len(e)), 'constant', constant_values=e[-1]) 
                                 for e in edc_fresh])
else:
    edc_fresh_padded = edc_fresh

# Compare EDCs numerically
print(f"\n5. Comparing EDC arrays")
edc_diff = edc_saved_padded - edc_fresh_padded
edc_abs_diff = np.abs(edc_diff)

print(f"   Max absolute difference: {np.max(edc_abs_diff):.10f} dB")
print(f"   Mean absolute difference: {np.mean(edc_abs_diff):.10f} dB")
print(f"   RMS difference: {np.sqrt(np.mean(edc_diff**2)):.10f} dB")

# Check if they're identical
if np.allclose(edc_saved_padded, edc_fresh_padded, atol=1e-10):
    print("   ✓ EDCs are IDENTICAL (within numerical precision)")
else:
    print("   ✗ EDCs are DIFFERENT!")

# Detailed comparison for test receiver
print(f"\n6. Detailed comparison for Receiver {TEST_RECEIVER_INDEX+1}")
rx, ry = receiver_positions[TEST_RECEIVER_INDEX]
print(f"   Position: ({rx:.2f}, {ry:.2f})")

edc_s = edc_saved_padded[TEST_RECEIVER_INDEX]
edc_f = edc_fresh_padded[TEST_RECEIVER_INDEX]
diff = edc_s - edc_f

print(f"   First 10 values (saved):  {edc_s[:10]}")
print(f"   First 10 values (fresh):  {edc_f[:10]}")
print(f"   First 10 differences:     {diff[:10]}")
print(f"   Last 10 values (saved):   {edc_s[-10:]}")
print(f"   Last 10 values (fresh):   {edc_f[-10:]}")
print(f"   Last 10 differences:      {diff[-10:]}")

# Calculate RMSE using both saved and fresh EDCs
print(f"\n7. Computing RMSE vs {REFERENCE_METHOD}")
print(f"   Using same reference EDCs for both comparisons")

rmse_saved_list = []
rmse_fresh_list = []

for i in range(len(receiver_positions)):
    # RMSE with saved EDC
    rmse_saved = an.compute_RMS(
        edc_saved[i], edc_ref[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_saved_list.append(rmse_saved)
    
    # RMSE with fresh EDC
    rmse_fresh = an.compute_RMS(
        edc_fresh[i], edc_ref[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_fresh_list.append(rmse_fresh)
    
    if i < 3 or i == TEST_RECEIVER_INDEX:
        print(f"   Receiver {i+1:2d}: Saved RMSE={rmse_saved:.6f}, Fresh RMSE={rmse_fresh:.6f}, Diff={rmse_fresh-rmse_saved:+.6f}")

mean_rmse_saved = np.mean(rmse_saved_list)
mean_rmse_fresh = np.mean(rmse_fresh_list)

print(f"\n8. FINAL RESULTS")
print("="*80)
print(f"Mean RMSE (using SAVED EDCs): {mean_rmse_saved:.6f}")
print(f"Mean RMSE (using FRESH EDCs): {mean_rmse_fresh:.6f}")
print(f"Difference: {mean_rmse_fresh - mean_rmse_saved:+.6f}")
print("="*80)

if abs(mean_rmse_saved - mean_rmse_fresh) < 0.001:
    print("\n✓ CONCLUSION: Saved and fresh EDCs produce nearly identical RMSEs")
    print("  The discrepancy must be elsewhere (different c value, different source, etc.)")
else:
    print(f"\n✗ CONCLUSION: Saved and fresh EDCs produce DIFFERENT RMSEs")
    print(f"  paper_figures_spatial.py value:  {mean_rmse_saved:.6f}")
    print(f"  optimisation_singleC.py value:   {mean_rmse_fresh:.6f}")
    print(f"  The saved data in the .npz file is outdated or was generated differently!")
