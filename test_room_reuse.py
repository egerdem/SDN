"""
Test if reusing the same room object affects RIR calculation
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import geometry
from rir_calculators import calculate_sdn_rir, rir_normalisation
from analysis import analysis as an

# Load data
DATA_FILE = 'results/paper_data/aes_room_spatial_edc_data_top_middle_source.npz'
with np.load(DATA_FILE, allow_pickle=True) as data:
    room_params = data['room_params'][0]
    source_pos = data['source_pos']
    receiver_positions = data['receiver_positions']
    Fs = int(data['Fs'])
    duration = float(data['duration'])
    ref_edcs = data['edcs_RIMPY-neg10']

# Setup
room_parameters = room_params.copy()
room_parameters['reflection'] = np.sqrt(1 - room_params['absorption'])

# TEST 1: Create fresh room for EACH receiver (like generate_paper_data.py does)
print("TEST 1: Creating FRESH room object for each receiver")
print("=" * 80)

edcs_fresh_room = []
for i in range(min(3, len(receiver_positions))):  # Just first 3 for speed
    rx, ry = receiver_positions[i]
    
    # Create NEW room for each iteration
    room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
    num_samples = int(Fs * duration)
    impulse = geometry.Source.generate_signal('dirac', num_samples)
    room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
    room.set_microphone(rx, ry, room_params['mic z'])
    room.wallAttenuation = [room_parameters['reflection']] * 6
    
    config = {
        'enabled': True,
        'flags': {'specular_source_injection': True, 'source_weighting': 2.998},
        'label': 'Test'
    }
    
    _, rir, _, _ = calculate_sdn_rir(room_parameters, "Test", room, duration, Fs, config)
    rir_norm = rir_normalisation(rir, room, Fs, normalize_to_first_impulse=True)['single_rir']
    edc, _, _ = an.compute_edc(rir_norm, Fs, plot=False)
    edcs_fresh_room.append(edc)
    
    rmse = an.compute_RMS(edc, ref_edcs[i], range=50, Fs=Fs, skip_initial_zeros=True, normalize_by_active_length=True)
    print(f"  Receiver {i+1}: RMSE = {rmse:.6f}")

mean_fresh = np.mean([an.compute_RMS(edcs_fresh_room[i], ref_edcs[i], range=50, Fs=Fs, skip_initial_zeros=True, normalize_by_active_length=True) for i in range(len(edcs_fresh_room))])

# TEST 2: Reuse SAME room object, just update mic position (like optimization does)
print("\nTEST 2: REUSING same room object, just updating mic position")
print("=" * 80)

# Create room ONCE
room_reused = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room_reused.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
room_reused.wallAttenuation = [room_parameters['reflection']] * 6

edcs_reused_room = []
for i in range(min(3, len(receiver_positions))):
    rx, ry = receiver_positions[i]
    
    # Just UPDATE mic position on same room
    room_reused.set_microphone(rx, ry, room_params['mic z'])
    
    config = {
        'enabled': True,
        'flags': {'specular_source_injection': True, 'source_weighting': 2.998},
        'label': 'Test'
    }
    
    _, rir, _, _ = calculate_sdn_rir(room_parameters, "Test", room_reused, duration, Fs, config)
    rir_norm = rir_normalisation(rir, room_reused, Fs, normalize_to_first_impulse=True)['single_rir']
    edc, _, _ = an.compute_edc(rir_norm, Fs, plot=False)
    edcs_reused_room.append(edc)
    
    rmse = an.compute_RMS(edc, ref_edcs[i], range=50, Fs=Fs, skip_initial_zeros=True, normalize_by_active_length=True)
    print(f"  Receiver {i+1}: RMSE = {rmse:.6f}")

mean_reused = np.mean([an.compute_RMS(edcs_reused_room[i], ref_edcs[i], range=50, Fs=Fs, skip_initial_zeros=True, normalize_by_active_length=True) for i in range(len(edcs_reused_room))])

# COMPARISON
print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)
print(f"Mean RMSE (fresh room each time): {mean_fresh:.6f}")
print(f"Mean RMSE (reused room):          {mean_reused:.6f}")
print(f"Difference:                        {abs(mean_fresh - mean_reused):.6f}")

if abs(mean_fresh - mean_reused) > 0.001:
    print("\n✗ ROOM OBJECT REUSE AFFECTS RESULTS!")
    print("   This could explain the discrepancy!")
    print("   generate_paper_data.py creates fresh room? optimization reuses room?")
else:
    print("\n✓ Room object reuse doesn't matter")
    print("   The discrepancy must be something else")
