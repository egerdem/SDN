"""
Inspect HO-SDN (Higher-Order SDN) after initialization but before processing samples.

HO-SDN differs from standard SDN by:
1. Early reflections (order < N) go directly to mic (bypass the scattering network)
2. Order-N reflections feed the SDN network
3. Additional delay lines for both cases
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
duration = 0.1  # Short duration
num_samples = int(Fs * duration)

# Create room geometry
room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
room.set_microphone(room_params['mic x'], room_params['mic y'], room_params['mic z'])
room.set_source(room_params['source x'], room_params['source y'], room_params['source z'],
                signal=np.zeros(num_samples), Fs=Fs)

# Set wall attenuation
room_params['reflection'] = np.sqrt(1 - room_params['absorption'])
room.wallAttenuation = [room_params['reflection']] * 6

print("=" * 100)
print("HO-SDN INITIALIZATION INSPECTION")
print("=" * 100)
print(f"\nRoom: {room_params['width']}m × {room_params['depth']}m × {room_params['height']}m")
print(f"Source: ({room_params['source x']}, {room_params['source y']}, {room_params['source z']})")
print(f"Mic: ({room_params['mic x']}, {room_params['mic y']}, {room_params['mic z']})")
print(f"Sampling rate: {Fs} Hz\n")

# ============================================================================
# Try different HO-SDN orders
# ============================================================================
for ho_order in [1, 2, 3]:
    print("\n" + "=" * 100)
    print(f"INITIALIZING HO-SDN with ORDER N = {ho_order}")
    print("=" * 100)
    
    sdn = DelayNetwork(
        room=room,
        Fs=Fs,
        c=343.0,
        ho_sdn_order=ho_order  # This activates HO-SDN mode
    )
    
    print(f"\n✅ HO-SDN Order-{ho_order} initialized (no samples processed yet)\n")
    
    # ========================================================================
    # INSPECT: Image Source Paths
    # ========================================================================
    print("-" * 100)
    print("1. IMAGE SOURCE PATHS (computed at initialization)")
    print("-" * 100)
    
    print(f"\n📍 Total paths up to order {ho_order}:")
    total_paths = len(sdn.early_reflection_paths) + len(sdn.sdn_feeding_paths)
    print(f"  Total: {total_paths} paths")
    print(f"  Early reflections (order < {ho_order}, direct to mic): {len(sdn.early_reflection_paths)}")
    print(f"  Order-{ho_order} reflections (feed SDN network): {len(sdn.sdn_feeding_paths)}")
    
    # Show some early reflection paths
    if len(sdn.early_reflection_paths) > 0:
        print(f"\n📍 Early reflection paths (first 5 of {len(sdn.early_reflection_paths)}):")
        for i, path_info in enumerate(sdn.early_reflection_paths[:5]):
            path_str = " → ".join(path_info['path'])
            dist = path_info['position'].getDistance(room.micPos)
            order = path_info['order']
            print(f"  Path {i}: [{path_str}] (order {order}, distance: {dist:.3f}m)")
    
    # Show some SDN-feeding paths
    if len(sdn.sdn_feeding_paths) > 0:
        print(f"\n📍 Order-{ho_order} paths feeding SDN (first 5 of {len(sdn.sdn_feeding_paths)}):")
        for i, path_info in enumerate(sdn.sdn_feeding_paths[:5]):
            path_str = " → ".join(path_info['path'])
            last_wall = path_info['path'][-1]
            dist_to_mic = path_info['position'].getDistance(room.micPos)
            print(f"  Path {i}: [{path_str}] → feeds '{last_wall}' node")
            print(f"    Image source to mic distance: {dist_to_mic:.3f}m")
    
    # ========================================================================
    # INSPECT: HO-SDN Specific Delay Lines
    # ========================================================================
    print("\n" + "-" * 100)
    print("2. HO-SDN SPECIFIC DELAY LINES")
    print("-" * 100)
    
    print("\n📍 Early reflection delay lines (order < N, bypass scattering):")
    if hasattr(sdn, 'early_reflection_del_lines') and sdn.early_reflection_del_lines:
        for i, (key, deque_obj) in enumerate(list(sdn.early_reflection_del_lines.items())[:5]):
            path_info = sdn.early_reflection_paths[i]
            dist = path_info['position'].getDistance(room.micPos)
            delay = len(deque_obj)
            print(f"  {key}: length={delay} samples")
            print(f"    → Path: {' → '.join(path_info['path'])}, order: {path_info['order']}")
            print(f"    → Distance: {dist:.3f}m, Travel time: {delay / Fs * 1000:.2f}ms")
        
        if len(sdn.early_reflection_del_lines) > 5:
            print(f"  ... ({len(sdn.early_reflection_del_lines) - 5} more early reflection lines)")
    else:
        print("  None (all reflections go through SDN network)")
    
    print(f"\n📍 Order-{ho_order} source-to-node delay lines (feed SDN network):")
    if hasattr(sdn, 'ho_source_to_nodes') and sdn.ho_source_to_nodes:
        # Group by destination wall
        paths_by_wall = {}
        for i, path_info in enumerate(sdn.sdn_feeding_paths):
            last_wall = path_info['path'][-1]
            if last_wall not in paths_by_wall:
                paths_by_wall[last_wall] = []
            paths_by_wall[last_wall].append((i, path_info))
        
        for wall_id, paths in paths_by_wall.items():
            print(f"\n  Paths feeding '{wall_id}' node: {len(paths)} paths")
            for i, path_info in paths[:2]:  # Show first 2 per wall
                key = f"sdn_feed_path_{i}_to_{wall_id}"
                if key in sdn.ho_source_to_nodes:
                    deque_obj = sdn.ho_source_to_nodes[key]
                    delay = len(deque_obj)
                    path_str = ' → '.join(path_info['path'])
                    
                    # Calculate alpha_k (delay from image source to node)
                    node_pos = room.walls[wall_id].node_positions
                    delta_true = path_info['position'].getDistance(room.micPos)
                    beta_k = node_pos.getDistance(room.micPos)
                    alpha_k = delta_true - beta_k
                    
                    print(f"    {key}:")
                    print(f"      Path: [{path_str}]")
                    print(f"      α_k (img_src to node): {alpha_k:.3f}m, delay: {delay} samples ({delay / Fs * 1000:.2f}ms)")
                    print(f"      β_k (node to mic): {beta_k:.3f}m")
                    print(f"      δ (img_src to mic): {delta_true:.3f}m")
            
            if len(paths) > 2:
                print(f"    ... ({len(paths) - 2} more paths to this wall)")
    else:
        print("  None (standard SDN mode)")
    
    # ========================================================================
    # INSPECT: Standard SDN Components (still present)
    # ========================================================================
    print("\n" + "-" * 100)
    print("3. STANDARD SDN COMPONENTS (also present in HO-SDN)")
    print("-" * 100)
    
    print("\n📍 Direct path:")
    print(f"  source_to_mic: delay={sdn.direct_sound_delay} samples ({sdn.direct_sound_delay / Fs * 1000:.2f}ms)")
    
    print(f"\n📍 Node-to-mic delay lines: {len(sdn.node_to_mic)}")
    for key in list(sdn.node_to_mic.keys())[:3]:
        delay = len(sdn.node_to_mic[key])
        print(f"  {key}: {delay} samples")
    
    print(f"\n📍 Node-to-node connections: {sdn.num_nodes * (sdn.num_nodes - 1)} total")
    
    # ========================================================================
    # INSPECT: Gain Factors (HO-SDN specific)
    # ========================================================================
    print("\n" + "-" * 100)
    print("4. GAIN FACTORS (HO-SDN uses different formulas)")
    print("-" * 100)
    
    if hasattr(sdn, 'early_reflection_gains') and sdn.early_reflection_gains:
        print(f"\n📍 Early reflection gains: {len(sdn.early_reflection_gains)}")
        for i, (key, gain) in enumerate(list(sdn.early_reflection_gains.items())[:5]):
            path_info = sdn.early_reflection_paths[i]
            dist = path_info['position'].getDistance(room.micPos)
            print(f"  {key}: {gain:.6f}")
            print(f"    Formula: G / distance = {343.0/Fs:.6f} / {dist:.3f} = {gain:.6f}")
        
        if len(sdn.early_reflection_gains) > 5:
            print(f"  ... ({len(sdn.early_reflection_gains) - 5} more)")
    
    if hasattr(sdn, 'ho_source_to_node_gains') and sdn.ho_source_to_node_gains:
        print(f"\n📍 Order-{ho_order} source-to-node gains: {len(sdn.ho_source_to_node_gains)}")
        for i, (key, gain) in enumerate(list(sdn.ho_source_to_node_gains.items())[:3]):
            print(f"  {key}: {gain:.6f}")
            # Extract path index
            path_idx = int(key.split('_')[3])
            if path_idx < len(sdn.sdn_feeding_paths):
                path_info = sdn.sdn_feeding_paths[path_idx]
                wall_id = path_info['path'][-1]
                node_pos = room.walls[wall_id].node_positions
                alpha = node_pos.getDistance(path_info['position'])
                beta = node_pos.getDistance(room.micPos)
                print(f"    α={alpha:.3f}m, β={beta:.3f}m")
                print(f"    Formula: G / (1 + α/β) = {343.0/Fs:.6f} / (1 + {alpha:.3f}/{beta:.3f}) = {gain:.6f}")
        
        if len(sdn.ho_source_to_node_gains) > 3:
            print(f"  ... ({len(sdn.ho_source_to_node_gains) - 3} more)")
    
    print(f"\n📍 Node-to-mic gains (HO-SDN formula: 1/β):")
    for key, gain in list(sdn.node_to_mic_gains.items())[:3]:
        wall_id = key.split('_to_')[0]
        beta = room.walls[wall_id].node_positions.getDistance(room.micPos)
        print(f"  {key}: {gain:.6f} (β={beta:.3f}m)")
    
    # ========================================================================
    # Summary for this order
    # ========================================================================
    print("\n" + "-" * 100)
    print(f"SUMMARY FOR ORDER-{ho_order}")
    print("-" * 100)
    print(f"""
Reflection Order Breakdown:
  • Order 0 (direct): 1 path → direct to mic
  • Orders 1 to {ho_order-1} (early): {len(sdn.early_reflection_paths)} paths → bypass SDN, direct to mic
  • Order {ho_order} (SDN-feeding): {len(sdn.sdn_feeding_paths)} paths → feed SDN network
  • Orders > {ho_order}: handled by SDN recursive scattering

Delay Lines Created:
  • Direct path: 1 line
  • Early reflections: {len(sdn.early_reflection_del_lines) if hasattr(sdn, 'early_reflection_del_lines') else 0} lines
  • Order-{ho_order} to nodes: {len(sdn.ho_source_to_nodes) if hasattr(sdn, 'ho_source_to_nodes') else 0} lines
  • Node to mic: {len(sdn.node_to_mic)} lines
  • Node to node: {sdn.num_nodes * (sdn.num_nodes - 1)} lines
  • TOTAL DELAY LINES: {1 + len(sdn.early_reflection_del_lines if hasattr(sdn, 'early_reflection_del_lines') else []) + len(sdn.ho_source_to_nodes if hasattr(sdn, 'ho_source_to_nodes') else []) + len(sdn.node_to_mic) + sdn.num_nodes * (sdn.num_nodes - 1)}

Key Insight:
  HO-SDN Order-{ho_order} treats reflections up to order {ho_order-1} as "early" (deterministic,
  like ISM), while using the SDN network to approximate orders ≥ {ho_order} recursively.
  This gives the accuracy of ISM for early reflections with the efficiency of SDN
  for late reverberation.
""")

# ============================================================================
# Final comparison table
# ============================================================================
print("\n" + "=" * 100)
print("COMPARISON: STANDARD SDN vs HO-SDN")
print("=" * 100)

# Reinitialize standard SDN for comparison
sdn_standard = DelayNetwork(room=room, Fs=Fs, c=343.0)

print(f"""
{'Feature':<40} {'Standard SDN':<20} {'HO-SDN (N=2)':<20} {'HO-SDN (N=3)':<20}
{'-'*100}
{'Early reflection handling':<40} {'All via network':<20} {'Order 1 direct':<20} {'Orders 1-2 direct':<20}
{'Source-to-node delay lines':<40} {f'{len(sdn_standard.source_to_nodes)} (1st order)':<20} {'Variable':<20} {'Variable':<20}
{'Early reflection lines':<40} {'0':<20} {'~6 (order 1)':<20} {'~30 (orders 1-2)':<20}
{'Computational cost':<40} {'Low':<20} {'Medium':<20} {'Higher':<20}
{'Early reflection accuracy':<40} {'Approximate':<20} {'Exact (order 1)':<20} {'Exact (orders 1-2)':<20}
{'Late reverberation':<40} {'Recursive':<20} {'Recursive':<20} {'Recursive':<20}

Key Tradeoff: Higher HO-SDN order → more accuracy → more delay lines → higher cost
""")

print("\n" + "=" * 100)
print("To see processing in action, run main.py with HO-SDN flags enabled:")
print("  RUN_MY_HO_SDN_n2_swc5 = True  # For Order-2 HO-SDN")
print("  RUN_MY_HO_SDN_n3_swc3 = True  # For Order-3 HO-SDN")
print("=" * 100 + "\n")

