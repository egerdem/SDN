"""
Test if calling calculate_sdn_rir_fast() modifies the room object
in a way that affects subsequent cache generation
"""
import numpy as np
import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import geometry
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation, calculate_sdn_rir
from analysis import analysis as an

print("=" * 90)
print("TESTING: Does calculate_sdn_rir_fast modify room state?")
print("=" * 90)

# Setup
DATA_FILE = 'results/paper_data/aes_room_spatial_edc_data_top_middle_source.npz'
with np.load(DATA_FILE, allow_pickle=True) as data:
    room_params = data['room_params'][0]
    source_pos = data['source_pos']
    rx, ry = data['receiver_positions'][0]  # Just first receiver
    Fs = int(data['Fs'])
    duration = float(data['duration'])

room_parameters = room_params.copy()
room_parameters['reflection'] = np.sqrt(1 - room_params['absorption'])

# Test 1: Create room, calculate with Fast method
print("\n=== Test 1: calculate_sdn_rir_fast ===")
room1 = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room1.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
room1.set_microphone(rx, ry, room_params['mic z'])
room1.wallAttenuation = [room_parameters['reflection']] * 6

print(f"BEFORE call:")
print(f"  room. wallAttenuation: {room1.wallAttenuation}")
print(f"  room.source: {room1.source}")
print(f"  room.micPos: {room1.micPos}")

config1 = {
    'flags': {'specular_source_injection': True, 'source_weighting': 2.998},
    'use_fast_method': True,
    'cache_label': 'test1'
}

# Clear cache first
from rir_calculators import _BASIS_CACHE, _CACHE_DIR
_BASIS_CACHE.clear()

_, rir1, _, _ = calculate_sdn_rir_fast(room_parameters, "Test", room1, duration, Fs, config1)

print(f"\nAFTER call:")
print(f"  room.wallAttenuation: {room1.wallAttenuation}")
print(f"  room.source: {room1.source}")
print(f"  room.micPos: {room1.micPos}")
print(f"  RIR length: {len(rir1)}")

# Test 2: Create FRESH room, calculate with Standard method, check if results match
print("\n\n=== Test 2: calculate_sdn_rir (Standard) ===")
room2 = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
impulse2 = geometry.Source.generate_signal('dirac', num_samples)
room2.set_source(*source_pos, signal=impulse2['signal'], Fs=Fs)
room2.set_microphone(rx, ry, room_params['mic z'])
room2.wallAttenuation = [room_parameters['reflection']] * 6

config2 = {
    'flags': {'specular_source_injection': True, 'source_weighting': 2.998},
    'label': 'Test-Standard'
}

_,rir2, _, _ = calculate_sdn_rir(room_parameters, "TestStd", room2, duration, Fs, config2)

print(f"  RIR length: {len(rir2)}")

# Compare
rir1_norm = rir_normalisation(rir1, room1, Fs, normalize_to_first_impulse=True)['single_rir']
rir2_norm = rir_normalisation(rir2, room2, Fs, normalize_to_first_impulse=True)['single_rir']

diff = rir1_norm - rir2_norm
rmse_diff = np.sqrt(np.mean(diff**2))
max_diff = np.max(np.abs(diff))

print(f"\n=== COMPARISON ===")
print(f"RMSE difference between Fast and Standard: {rmse_diff:.10e}")
print(f"Max difference: {max_diff:.10e}")

if rmse_diff < 1e-10:
    print("✓ IDENTICAL - Fast and Standard produce same RIR")
else:
    print(f"✗ DIFFERENT - Fast and Standard RIRs differ by {rmse_diff}")

#Test 3: What if we call Standard BEFORE Fast? Does it affect the cache?
print("\n\n=== Test 3: Standard call, THEN Fast call ===")
_BASIS_CACHE.clear()

room3 = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
impulse3 = geometry.Source.generate_signal('dirac', num_samples)
room3.set_source(*source_pos, signal=impulse3['signal'], Fs=Fs)
room3.set_microphone(rx, ry, room_params['mic z'])
room3.wallAttenuation = [room_parameters['reflection']] * 6

# First call Standard (this gets called by Fast internally to create basis)
print("Calling Standard method first...")
_, rir_std_first, _, _ = calculate_sdn_rir(room_parameters, "PreCall", room3, duration, Fs, config2)

# Now call Fast (will it reuse something from the previous call?)
print("Now calling Fast method...")
_, rir_fast_after, _, _ = calculate_sdn_rir_fast(room_parameters, "Test", room3, duration, Fs, config1)

rir_fast_after_norm = rir_normalisation(rir_fast_after, room3, Fs, normalize_to_first_impulse=True)['single_rir']
diff2 = rir1_norm - rir_fast_after_norm
rmse_diff2 = np.sqrt(np.mean(diff2**2))

print(f"RMSE difference vs original Fast call: {rmse_diff2:.10e}")

if rmse_diff2 > 1e-10:
    print(f"✗ ORDER MATTERS! Calling Standard before Fast changes the result!")
else:
    print("✓ Order doesn't matter, results identical")

print("\n=== CONCLUSION ===")
print("If order matters or Fast ≠ Standard, then room object state is being")
print("modified in a way that affects the cached basis functions!")
