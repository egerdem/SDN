
import sys
import os
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

import experiment_configs as exp_config
try:
    from analysis.spatial_analysis import generate_source_positions, generate_full_receiver_grid
except ImportError:
    from spatial_analysis import generate_source_positions, generate_full_receiver_grid

def harmonic_mean(dists):
    """
    Harmonic Mean = n / sum(1/x_i).
    Dominated by smallest values (proximity to any wall).
    """
    inv_sum = sum(1.0 / d for d in dists if d > 0)
    if inv_sum == 0: return float('inf')
    return len(dists) / inv_sum

def get_wall_distances(x, y, z, width, depth, height):
    """Calculate distances to all 6 walls."""
    return [
        x,              # x=0 (West)
        width - x,      # x=W (East)
        y,              # y=0 (South)
        depth - y,      # y=D (North)
        z,              # z=0 (Floor)
        height - z      # z=H (Ceiling)
    ]

def analyze_wall_proximity():
    # Load Room Config
    active_room = exp_config.active_room
    width = active_room['width']
    depth = active_room['depth']
    height = active_room['height']
    
    room_name = active_room.get('display_name', 'Unknown')
    print(f"Room: {room_name} (W={width}, D={depth}, H={height})")
    print("\n--- Harmonic Mean Explanation ---")
    print("Harmonic Mean (H) is defined as n / sum(1/x_i).")
    print("It is dominated by the SMALLEST distance.")
    print("Example: [10, 10, 10] -> H=10.0")
    print("Example: [10, 10, 0.1] -> H=0.29 (Shows 'dangerous' proximity despite other walls being far)")
    
import geometry

def analyze_wall_proximity():
    # Load Room Config
    active_room = exp_config.active_room
    width = active_room['width']
    depth = active_room['depth']
    height = active_room['height']
    
    room_name = active_room.get('display_name', 'Unknown')
    print(f"Room: {room_name} (W={width}, D={depth}, H={height})")
    
    # Generate Sources and Receivers
    source_positions = generate_source_positions(active_room, name='v1')
    receiver_positions = generate_full_receiver_grid(width, depth, active_room['mic z'], 4, 4, 0.5)
    
    # -------------------------------------------------------------------------
    # 1. Source Summary (Harmonic Mean)
    # -------------------------------------------------------------------------
    print(f"\n[1. SOURCE Proximity Summary]")
    print(f"{'Source Name':<20} | {'Coords':<20} | {'H_src':<10}")
    print("-" * 60)
    
    src_h_means = {}
    
    for src in source_positions:
        sx, sy, sz, name = src
        dists = get_wall_distances(sx, sy, sz, width, depth, height)
        h_src = harmonic_mean(dists)
        src_h_means[name] = h_src
        print(f"{name:<20} | {f'({sx:.1f}, {sy:.1f})':<20} | {h_src:<10.3f}")

    # -------------------------------------------------------------------------
    # 2. Detailed Grid Analysis
    # -------------------------------------------------------------------------
    print(f"\n[2. Detailed Metric Analysis]")
    
    # We will store results to print tables later
    # tables['combined_score'][rx_idx][src_name] = score
    tables = {
        'combined': {},
        'node_dist': {}
    }
    
    for rx_idx, rx in enumerate(receiver_positions):
        rx_x, rx_y, rx_z = rx
        dists_rx = get_wall_distances(rx_x, rx_y, rx_z, width, depth, height)
        h_rcv = harmonic_mean(dists_rx)
        
        tables['combined'][rx_idx] = {'coords': (rx_x, rx_y)}
        tables['node_dist'][rx_idx] = {'coords': (rx_x, rx_y)}
        
        for src in source_positions:
            sx, sy, sz, src_name = src
            h_src = src_h_means[src_name]
            
            # Metric 1: Combined Proximity Score
            combined_score = (1.0 / h_src) + (1.0 / h_rcv)
            tables['combined'][rx_idx][src_name] = combined_score
            
            # Metric 2: Mean Distance to 3 Nearest Nodes
            # We must set up the room to calculate nodes
            room = geometry.Room(width, depth, height)
            room.set_source(sx, sy, sz)
            room.set_microphone(rx_x, rx_y, rx_z)
            # Nodes are calculated automatically in set_microphone if source is set
            
            node_dists = []
            if room.sdn_nodes_calculated:
                for wall in room.walls.values():
                    # Distance from Source -> Node (Image Source Projection on Wall)
                    # Note: room.walls[id].node_positions is the intersection point on the wall
                    # The distance we want is Source -> Point On Wall
                    np_pos = wall.node_positions
                    d = room.source.srcPos.getDistance(np_pos)
                    node_dists.append(d)
                
                node_dists.sort()
                mean_3_nodes = sum(node_dists[:3]) / 3.0
                tables['node_dist'][rx_idx][src_name] = mean_3_nodes
            else:
                 tables['node_dist'][rx_idx][src_name] = 0.0

    # Helper to print table
    source_names = [s[3] for s in source_positions]
    
    def print_metric_table(metric_key, title):
        print(f"\n{title}")
        print("-" * 100)
        
        header = f"{'Receiver':<10} | {'Coords':<12} ||"
        for name in source_names:
             header += f" {name[:10]:<10} |"
        print(header)
        print("-" * 100)
        
        col_sums = {name: 0.0 for name in source_names}
        
        for rx_idx in sorted(tables[metric_key].keys()):
            data = tables[metric_key][rx_idx]
            rx, ry = data['coords']
            row_str = f"R{rx_idx+1:<9} | {f'({rx:.1f},{ry:.1f})':<12} ||"
            
            for src_name in source_names:
                val = data.get(src_name, 0.0)
                col_sums[src_name] += val
                row_str += f" {val:<10.3f} |"
            print(row_str)
            
        print("-" * 100)
        # Mean Row
        mean_str = f"{'MEAN':<10} | {'-':<12} ||"
        for src_name in source_names:
            avg = col_sums[src_name] / len(receiver_positions)
            mean_str += f" {avg:<10.3f} |"
        print(mean_str)
        print("-" * 100)

    print_metric_table('combined', "Combined Proximity Score (1/H_src + 1/H_rcv) [Higher = Harder]")
    print_metric_table('node_dist', "Mean Distance to 3 Nearest Nodes [Lower = Harder]")

if __name__ == "__main__":
    analyze_wall_proximity()
