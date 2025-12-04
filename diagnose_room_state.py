"""
Diagnostic: Print room state to see what changes between calculations
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
    receiver_positions = data['receiver_positions'][:3]  # Just first 3
    Fs = int(data['Fs'])
    duration = float(data['duration'])

room_parameters = room_params.copy()
room_parameters['reflection'] = np.sqrt(1 - room_params['absorption'])

def print_room_state(room, label):
    """Print relevant room object state"""
    print(f"\n{label}:")
    print(f"  Source pos: ({room.source.srcPos.x:.4f}, {room.source.srcPos.y:.4f}, {room.source.srcPos.z:.4f})")
    
    if room.micPos is not None:
        print(f"  Mic pos: ({room.micPos.x:.4f}, {room.micPos.y:.4f}, {room.micPos.z:.4f})")
    else:
        print(f"  Mic pos: None")
        
    print(f"  Wall attenuation: {room.wallAttenuation}")
    print(f"  Room dimensions: {room.width}x{room.depth}x{room.height}")
    
    # Check if source signal changes
    if hasattr(room.source, 'signal'):
        print(f"  Source signal len: {len(room.source.signal) if room.source.signal is not None else 'None'}")
        if room.source.signal is not None:
            print(f"  Source signal sum: {np.sum(room.source.signal):.6f}")

print("="*80)
print("REUSING SAME ROOM OBJECT - Tracking state changes")
print("="*80)

# Create room ONCE
room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)
room.wallAttenuation = [room_parameters['reflection']] * 6

print_room_state(room, "INITIAL STATE")

config = {
    'enabled': True,
    'flags': {'specular_source_injection': True, 'source_weighting': 2.998},
    'label': 'Test'
}

for i, (rx, ry) in enumerate(receiver_positions):
    print(f"\n{'='*80}")
    print(f"RECEIVER {i+1} at ({rx:.2f}, {ry:.2f})")
    print(f"{'='*80}")
    
    print_room_state(room, f"BEFORE set_microphone")
    
    room.set_microphone(rx, ry, room_params['mic z'])
    
    print_room_state(room, f"AFTER set_microphone")
    
    _, rir, _, _ = calculate_sdn_rir(room_parameters, "Test", room, duration, Fs, config)
    
    print_room_state(room, f"AFTER calculate_sdn_rir")
    
    print(f"\n  RIR length: {len(rir)}")
    print(f"  RIR sum: {np.sum(rir):.6f}")
    print(f"  RIR max: {np.max(np.abs(rir)):.6f}")

print("\n" + "="*80)
print("KEY OBSERVATIONS:")
print("="*80)
print("Look for any changes to room state that persist across iterations.")
print("Common culprits:")
print("  - Source signal getting modified")
print("  - Wall attenuation changing")
print("  - Some internal cache or state not being reset")
