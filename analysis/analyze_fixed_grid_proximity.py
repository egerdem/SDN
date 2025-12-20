
import os
import sys
import numpy as np
# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)
if script_dir not in sys.path: sys.path.append(script_dir)

import experiment_configs
import geometry
from deprecated import wall_proximity
from c_predictor import predict_c_from_source

try:
    from analysis.spatial_analysis import generate_source_positions, generate_full_receiver_grid
except ImportError:
    from spatial_analysis import generate_source_positions, generate_full_receiver_grid, generate_receiver_grid_old

def analyze_fixed_grids():
    # Define Rooms to Analyze
    rooms_to_analyze = [
        # experiment_configs.room_cube6,
        experiment_configs.room_aes
    ]
    
    print("=== FIXED GRID PROXIMITY ANALYSIS ===")
    print("Metrics: Harmonic Mean (H_src), Combined Proximity Score, Mean 3-Nearest Node Dist")
    
    for active_room in rooms_to_analyze:
        room_name = active_room['display_name']
        w = active_room['width']
        d = active_room['depth']
        h = active_room['height']
        
        print(f"\n\nRoom: {room_name} (W={w}, D={d}, H={h})")
        print("=" * 100)
        
        # 1. Generate Sources (4 fixed positions)
        # Using name='v1' which corresponds to Center, Top-Mid, Upper-Right, Lower-Left
        # source_positions = generate_source_positions(active_room, name='v1')
        source_positions = generate_source_positions(active_room, name='v3')
        source_names = [s[3] for s in source_positions]
        
        # 2. Generate Receivers (16 fixed positions - Full Grid 4x4)
        # receiver_positions = generate_full_receiver_grid(w, d, active_room['mic z'], 4, 4, 0.5)
        receiver_positions = generate_receiver_grid_old(active_room['width'] / 2, active_room['depth'] / 2, z = active_room['mic z'], margin=0.5,
                                                        n_points=16)

        # Data structure for tables
        # tables['metric'][rx_idx][src_name]
        metrics = {
            'combined_score': {},
            'node_dist': {},
            'node_vol': {}, # Determinant of Covariance (Volume)
            'node_trace': {}, # Trace of Covariance (Total Spread)
            'h_src': {}
        }
        
        # Pre-calculate Source Metrics
        room_dims = (w, d, h)
        for src in source_positions:
            sx, sy, sz, s_name = src
            s_dists = wall_proximity.get_wall_distances(sx, sy, sz, w, d, h)
            h_s = wall_proximity.harmonic_mean(s_dists)
            metrics['h_src'][s_name] = h_s
            
        print(f"Sources H_src: {', '.join([f'{k}={v:.2f}' for k,v in metrics['h_src'].items()])}")

        # Print C predictions for each source using the fitted linear rule (via c_predictor.py)
        print("\nSuggested C (from linear fit) per source:")
        print(f"{'Source':<24} | {'Pos (x,y,z) [m]':<22} | {'H_src':<8} | {'C_pred':<6}")
        print("-" * 70)
        for sx, sy, sz, s_name in source_positions:
            h_s = metrics['h_src'][s_name]
            c_pred = predict_c_from_source((sx, sy, sz), room_dims, model_type="linear", verbose=False)
            pos_str = f"({sx:.2f},{sy:.2f},{sz:.2f})"
            print(f"{s_name:<24} | {pos_str:<22} | {h_s:<8.3f} | {c_pred:<6.2f}")
        
        # Calculate Metrics for all pairs
        for rx_idx, rx in enumerate(receiver_positions):
            rx_x, rx_y, rx_z = rx
            dists_rx = wall_proximity.get_wall_distances(rx_x, rx_y, rx_z, w, d, h)
            h_rcv = wall_proximity.harmonic_mean(dists_rx)
            
            metrics['combined_score'][rx_idx] = {'coords': (rx_x, rx_y)}
            metrics['node_dist'][rx_idx] = {'coords': (rx_x, rx_y)}
            metrics['node_vol'][rx_idx] = {'coords': (rx_x, rx_y)}
            metrics['node_trace'][rx_idx] = {'coords': (rx_x, rx_y)}
            
            for src in source_positions:
                sx, sy, sz, src_name = src
                h_s = metrics['h_src'][src_name]
                
                # Metric 1: Combined Score
                score = (1.0 / h_s) + (1.0 / h_rcv)
                metrics['combined_score'][rx_idx][src_name] = score
                
                # Metric 2: Mean 3-Nearest Node Distance & Eigenvalues
                temp_room = geometry.Room(w, d, h)
                temp_room.set_source(sx, sy, sz) 
                temp_room.set_microphone(rx_x, rx_y, rx_z)
                
                node_positions = []
                if temp_room.sdn_nodes_calculated:
                    # 1. Collect Node Positions
                    for wall in temp_room.walls.values():
                        np_pos = wall.node_positions
                        node_positions.append([np_pos.x, np_pos.y, np_pos.z])
                    
                    # 2. Node Distance (Mean 3-Nearest)
                    node_dists = []
                    for np_pos_list in node_positions:
                        dist = temp_room.source.srcPos.getDistance(geometry.Point(*np_pos_list))
                        node_dists.append(dist)
                    node_dists.sort()
                    mean_node = sum(node_dists[:3]) / 3.0
                    metrics['node_dist'][rx_idx][src_name] = mean_node
                    
                    # 3. Eigenvalue Analysis (Covariance Matrix)
                    # Shape: (3, N)
                    points = np.array(node_positions).T 
                    cov_matrix = np.cov(points)
                    eigvals = np.linalg.eigvalsh(cov_matrix)
                    
                    # Metrics
                    trace = np.sum(eigvals) # Total Variance (Spread)
                    vol = np.prod(eigvals)  # Volume (Determinant)
                    # Use Volume simply
                    metrics['node_vol'][rx_idx][src_name] = vol
                    metrics['node_trace'][rx_idx][src_name] = trace

                else:
                    metrics['node_dist'][rx_idx][src_name] = 0.0
                    metrics['node_vol'][rx_idx][src_name] = 0.0
                    metrics['node_trace'][rx_idx][src_name] = 0.0

        # Print Tables
        print_metric_table(metrics['combined_score'], source_names, receiver_positions, "Combined Proximity Score (Higher = More Confined)")
        print_metric_table(metrics['node_dist'], source_names, receiver_positions, "Mean Distance to 3 Nearest Nodes (Lower = Images Bunched)")
        print_metric_table(metrics['node_trace'], source_names, receiver_positions, "Node Cloud Total Spread (Trace of Cov) [Lower = Bunched]")
        print_metric_table(metrics['node_vol'], source_names, receiver_positions, "Node Cloud Volume (Det of Cov) [Lower = Flat/Bunched]")



def print_metric_table(data_dict, source_names, receiver_positions, title):
    print(f"\n{title}")
    print("-" * 120)
    
    header = f"{'Recv':<8} | {'Coords':<12} ||"
    for name in source_names:
         # Shorten name
         short_name = name.replace("Source", "").replace("_", " ").strip()
         header += f" {short_name[:12]:<12} |"
    print(header)
    print("-" * 120)
    
    col_sums = {name: 0.0 for name in source_names}
    count = len(data_dict)
    
    for rx_idx in sorted(data_dict.keys()):
        data = data_dict[rx_idx]
        rx, ry = data['coords']
        row_str = f"R{rx_idx+1:<7} | ({rx:<4.1f}, {ry:<4.1f})   ||"
        
        for src_name in source_names:
            val = data.get(src_name, 0.0)
            col_sums[src_name] += val
            row_str += f" {val:<12.3f} |"
        print(row_str)
        
    print("-" * 120)
    # Mean Row
    mean_str = f"{'MEAN':<8} | {'-':<12} ||"
    for src_name in source_names:
        avg = col_sums[src_name] / count
        mean_str += f" {avg:<12.3f} |"
    print(mean_str)
    print("-" * 120)

if __name__ == "__main__":
    output_path = os.path.join(project_root, "results", "fixed_grid_proximity_results.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Keep original stdout to restore/print completion message
    original_stdout = sys.stdout
    
    with open(output_path, "w") as f:
        sys.stdout = f
        try:
            analyze_fixed_grids()
        finally:
            sys.stdout = original_stdout
            
    print(f"Analysis complete. Results saved to: {output_path}")
