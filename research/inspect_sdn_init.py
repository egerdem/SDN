"""
Inspect DelayNetwork after initialization but before processing any samples.
This helps understand what data structures are created at __init__ vs during processing.
"""

import numpy as np
import geometry
from sdn_core import DelayNetwork
from pprint import pprint

# ============================================================================
# Setup a simple room
# ============================================================================
room_params = {
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 2, 'mic y': 2, 'mic z': 1.5,
    'absorption': 0.2,
}

Fs = 44100
duration = 0.1  # Short duration, we won't actually process
num_samples = int(Fs * duration)

# Create room geometry
room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
room.set_microphone(room_params['mic x'], room_params['mic y'], room_params['mic z'])
room.set_source(room_params['source x'], room_params['source y'], room_params['source z'],
                signal=np.zeros(num_samples), Fs=Fs)

# Set wall attenuation
room_params['reflection'] = np.sqrt(1 - room_params['absorption'])
room.wallAttenuation = [room_params['reflection']] * 6

print("=" * 80)
print("SDN INITIALIZATION INSPECTION")
print("=" * 80)
print(f"\nRoom: {room_params['width']}m × {room_params['depth']}m × {room_params['height']}m")
print(f"Source: ({room_params['source x']}, {room_params['source y']}, {room_params['source z']})")
print(f"Mic: ({room_params['mic x']}, {room_params['mic y']}, {room_params['mic z']})")
print(f"Absorption: {room_params['absorption']}, Reflection: {room_params['reflection']:.3f}")
print(f"Sampling rate: {Fs} Hz")

# ============================================================================
# Initialize DelayNetwork (but don't process any samples yet)
# ============================================================================
print("\n" + "=" * 80)
print("INITIALIZING DelayNetwork...")
print("=" * 80)

sdn = DelayNetwork(
    room=room,
    Fs=Fs,
    c=343.0,
    specular_source_injection=True,  # Example flag
    source_weighting=3
)

print("\n✅ DelayNetwork initialized (no samples processed yet)\n")

# ============================================================================
# INSPECT: Delay Lines
# ============================================================================
print("=" * 80)
print("1. DELAY LINES (deques created at initialization)")
print("=" * 80)

print("\n📍 Direct path (source → mic):")
for key, deque_obj in sdn.source_to_mic.items():
    print(f"  {key}: length={len(deque_obj)} samples, delay={sdn.direct_sound_delay} samples")
    print(f"    → Physical distance: {room.source.srcPos.getDistance(room.micPos):.3f}m")
    print(f"    → Travel time: {sdn.direct_sound_delay / Fs * 1000:.2f}ms")

print("\n📍 Source to nodes (1st order reflections):")
for key, deque_obj in sdn.source_to_nodes.items():
    wall_id = key.split('_to_')[1]
    delay = getattr(sdn, f'src_to_{wall_id}_delay')
    dist = room.walls[wall_id].node_positions.getDistance(room.source.srcPos)
    print(f"  {key}: length={len(deque_obj)} samples, delay={delay}")
    print(f"    → Distance: {dist:.3f}m, Travel time: {delay / Fs * 1000:.2f}ms")

print("\n📍 Nodes to mic:")
for key, deque_obj in sdn.node_to_mic.items():
    wall_id = key.split('_to_')[0]
    delay = getattr(sdn, f'{wall_id}_to_mic_delay')
    dist = room.walls[wall_id].node_positions.getDistance(room.micPos)
    print(f"  {key}: length={len(deque_obj)} samples, delay={delay}")
    print(f"    → Distance: {dist:.3f}m, Travel time: {delay / Fs * 1000:.2f}ms")

print("\n📍 Node to node connections (sample: first 3):")
count = 0
for wall1_id, connections in sdn.node_to_node.items():
    for wall2_id, deque_obj in connections.items():
        if count < 3:
            delay = getattr(sdn, f'{wall1_id}_to_{wall2_id}_delay')
            dist = room.walls[wall1_id].node_positions.getDistance(room.walls[wall2_id].node_positions)
            print(f"  {wall1_id} → {wall2_id}: length={len(deque_obj)} samples, delay={delay}")
            print(f"    → Distance: {dist:.3f}m, Travel time: {delay / Fs * 1000:.2f}ms")
            count += 1
print(f"  ... (total {sdn.num_nodes * (sdn.num_nodes - 1)} connections)")

# ============================================================================
# INSPECT: Gain Factors
# ============================================================================
print("\n" + "=" * 80)
print("2. GAIN FACTORS (attenuation multipliers calculated at initialization)")
print("=" * 80)

print(f"\n📍 Direct path gain: {sdn.source_to_mic_gain:.6f}")

print("\n📍 Source to node gains:")
for key, gain in sdn.source_to_node_gains.items():
    print(f"  {key}: {gain:.6f}")

print("\n📍 Node to mic gains:")
for key, gain in sdn.node_to_mic_gains.items():
    print(f"  {key}: {gain:.6f}")

# ============================================================================
# INSPECT: Scattering Matrix
# ============================================================================
print("\n" + "=" * 80)
print("3. SCATTERING MATRIX (created at initialization)")
print("=" * 80)
print(f"\nMatrix shape: {sdn.scattering_matrix.shape}")
print("Matrix values:")
print(sdn.scattering_matrix)

# ============================================================================
# INSPECT: State Variables
# ============================================================================
print("\n" + "=" * 80)
print("4. STATE VARIABLES (initialized to zero, updated during processing)")
print("=" * 80)

print("\n📍 Node pressures (all zeros initially):")
pprint(sdn.node_pressures)
print("\n📍 Outgoing waves (all zeros initially, sample: first wall):")
first_wall = list(room.walls.keys())[0]
print(f"{first_wall} outgoing waves:")
pprint(sdn.outgoing_waves[first_wall])

# ============================================================================
# INSPECT: What's NOT created yet
# ============================================================================
print("\n" + "=" * 80)
print("5. WHAT HAPPENS DURING process_sample() (NOT YET EXECUTED)")
print("=" * 80)
print("""
The following happens per sample when process_sample(input_sample, n) is called:

1. Input sample is injected into delay lines (append operations)
2. Delay line outputs are read (pop left from deque index [0])
3. Scattering matrix is applied to incoming waves
4. State variables (node_pressures, outgoing_waves) are updated
5. New outgoing waves are appended to node_to_node delay lines

Current state: we initialized SDNNetwork class but didn't process any samples yet. 
            All delay lines contain only zeros
              State variables are all zeros
              Nothing has propagated yet.
""")

# ============================================================================
# OPTIONAL: Inspect a single delay line contents
# ============================================================================
print("\n" + "=" * 80)
print("6. EXAMPLE: Contents of one delay line (all zeros initially)")
print("=" * 80)
example_key = "src_to_floor"
if example_key in sdn.source_to_nodes:
    deque_contents = list(sdn.source_to_nodes[example_key])
    print(f"\n{example_key} delay line contents (first 10 samples):")
    print(deque_contents[:10])
    print(f"All values are zero: {all(x == 0 for x in deque_contents)}")
    print(f"Total length: {len(deque_contents)} samples")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: What's created at initialization vs during processing")
print("=" * 80)
print("""
AT INITIALIZATION (__init__):
  ✅ Delay lines (deques) - created with correct lengths, filled with zeros
  ✅ Gain factors - calculated from geometry
  ✅ Scattering matrix - created based on flags
  ✅ State variables - initialized to zero
  ✅ Room geometry and node positions

DURING PROCESSING (process_sample):
  🔄 Delay line contents - updated each sample (append/pop operations)
  🔄 State variables - updated each sample (node_pressures, outgoing_waves)
  🔄 Output sample - accumulated from multiple paths
  
NEVER CHANGES:
  🔒 Delay line lengths
  🔒 Gain factors
  🔒 Scattering matrix (unless using special experimental flags)
""")

print("\n" + "=" * 80)

# the next step is to propagate samples:
# Example: Process first 5 samples with an impulse:
# impulse = np.array([1.0] + [0.0] * 4)
# rir = np.zeros(5)
# for n in range(5):
#     rir[n] = sdn.process_sample(impulse[n], n)
#     print(f"Sample {n}: input={impulse[n]:.1f}, output={rir[n]:.6f}")


