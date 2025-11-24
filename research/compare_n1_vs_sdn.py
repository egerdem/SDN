"""
Compare HO-SDN N=1 with Standard SDN to verify they're equivalent.
"""
import numpy as np
import geometry
from sdn_core import DelayNetwork

# Setup room
room_params = {
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 2, 'mic y': 2, 'mic z': 1.5,
    'absorption': 0.2,
}

Fs = 44100
duration = 0.1
num_samples = int(Fs * duration)

# Create room
room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
room.set_microphone(room_params['mic x'], room_params['mic y'], room_params['mic z'])
room.set_source(room_params['source x'], room_params['source y'], room_params['source z'],
                signal=np.zeros(num_samples), Fs=Fs)

room_params['reflection'] = np.sqrt(1 - room_params['absorption'])
room.wallAttenuation = [room_params['reflection']] * 6

print("=" * 80)
print("COMPARING HO-SDN N=1 vs STANDARD SDN")
print("=" * 80)

# Create standard SDN
print("\n--- Standard SDN ---")
sdn_standard = DelayNetwork(room, Fs=Fs, c=343.0, source_weighting=1, specular_source_injection=True)

print("\nStandard SDN Gains:")
for wall_id in room.walls:
    src_key = f"src_to_{wall_id}"
    mic_key = f"{wall_id}_to_mic"
    g_sk = sdn_standard.source_to_node_gains[src_key]
    g_km = sdn_standard.node_to_mic_gains[mic_key]
    total = g_sk * g_km
    print(f"{wall_id:8}: g_Sk={g_sk:.6f}, g_kM={g_km:.6f}, total={total:.6f}")

# Create HO-SDN with N=1
print("\n--- HO-SDN with N=1 ---")
sdn_ho_n1 = DelayNetwork(room, Fs=Fs, c=343.0, ho_sdn_order=1)

print("\nHO-SDN N=1 Gains:")
print(f"Number of SDN-feeding paths: {len(sdn_ho_n1.sdn_feeding_paths)}")
print(f"Number of early reflection paths: {len(sdn_ho_n1.early_reflection_paths)}")

# Group by wall
gains_by_wall = {}
for i, img_info in enumerate(sdn_ho_n1.sdn_feeding_paths):
    wall_id = img_info['path'][-1]
    key = f"sdn_feed_path_{i}_to_{wall_id}"
    g_sk = sdn_ho_n1.ho_source_to_node_gains[key]
    mic_key = f"{wall_id}_to_mic"
    g_km = sdn_ho_n1.node_to_mic_gains[mic_key]
    total = g_sk * g_km
    
    if wall_id not in gains_by_wall:
        gains_by_wall[wall_id] = []
    gains_by_wall[wall_id].append((g_sk, g_km, total))

for wall_id in sorted(gains_by_wall.keys()):
    gains_list = gains_by_wall[wall_id]
    print(f"{wall_id:8}: {len(gains_list)} path(s)")
    for idx, (g_sk, g_km, total) in enumerate(gains_list):
        print(f"  Path {idx}: g_Sk={g_sk:.6f}, g_kM={g_km:.6f}, total={total:.6f}")

# Compare gains
print("\n" + "=" * 80)
print("GAIN COMPARISON:")
print("=" * 80)

for wall_id in room.walls:
    std_src_key = f"src_to_{wall_id}"
    std_mic_key = f"{wall_id}_to_mic"
    std_g_sk = sdn_standard.source_to_node_gains[std_src_key]
    std_g_km = sdn_standard.node_to_mic_gains[std_mic_key]
    std_total = std_g_sk * std_g_km
    
    if wall_id in gains_by_wall:
        ho_gains = gains_by_wall[wall_id][0]  # Should be only one path per wall for N=1
        ho_g_sk, ho_g_km, ho_total = ho_gains
        
        print(f"\n{wall_id}:")
        print(f"  Standard: g_Sk={std_g_sk:.6f}, g_kM={std_g_km:.6f}, total={std_total:.6f}")
        print(f"  HO-SDN N1: g_Sk={ho_g_sk:.6f}, g_kM={ho_g_km:.6f}, total={ho_total:.6f}")
        print(f"  Difference: g_Sk={abs(std_g_sk-ho_g_sk):.6e}, g_kM={abs(std_g_km-ho_g_km):.6e}, total={abs(std_total-ho_total):.6e}")
        
        if abs(std_total - ho_total) > 1e-6:
            print(f"  ⚠️ WARNING: Total gain mismatch!")

# Compare node positions
print("\n" + "=" * 80)
print("NODE POSITION COMPARISON:")
print("=" * 80)

for wall_id in room.walls:
    std_node = room.walls[wall_id].node_positions
    ho_node = sdn_ho_n1.ho_node_positions[wall_id]
    
    diff_x = abs(std_node.x - ho_node.x)
    diff_y = abs(std_node.y - ho_node.y)
    diff_z = abs(std_node.z - ho_node.z)
    total_diff = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    
    print(f"{wall_id:8}: Standard=({std_node.x:.4f}, {std_node.y:.4f}, {std_node.z:.4f}), "
          f"HO=({ho_node.x:.4f}, {ho_node.y:.4f}, {ho_node.z:.4f}), "
          f"diff={total_diff:.6e}")
    
    if total_diff > 1e-6:
        print(f"  ⚠️ WARNING: Node position mismatch!")

