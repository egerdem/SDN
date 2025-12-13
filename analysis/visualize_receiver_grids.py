"""
Visualize receiver grids for multiple source positions.

This script generates and displays receiver grids for each source position,
showing the spatial setup without running any experiments.
Also calculates and prints paths up to order 1 for all source-receiver combinations.
"""

import sys
import os

# Add script directory (analysis/) and project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if script_dir not in sys.path: sys.path.append(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

# Now standard imports should work
from spatial_analysis import (
    print_receiver_grid,
    generate_receiver_grid_old,
    generate_source_positions, generate_full_receiver_grid
)
import geometry
from sdn_path_calculator import SDNCalculator
import path_tracker
import experiment_configs as exp_config

if __name__ == "__main__":
    # Room parameters
    active_room = exp_config.active_room
    room_name = active_room.get('display_name')
    print(f"Selected room: {room_name}")

    GRID_SELECTION = "full" # "full" or "quarter"

    if GRID_SELECTION == "full":
        
        receiver_positions = generate_full_receiver_grid(
        room_width=active_room['width'],
        room_depth=active_room['depth'],
        height=active_room['mic z'],
        n_x=4,
        n_y=4,
        margin=0.5
        )
        grid_tag = "FULLGRID"
        print(f"Running in MULTI-POSITION (FULL-GRID) mode for room '{active_room['display_name']}'")

    elif GRID_SELECTION == "quarter":
        receiver_positions = generate_receiver_grid_old(
            active_room['width'] / 2,
            active_room['depth'] / 2,
            n_points=16,
            margin=0.5
        )
        print(f"Running in MULTI-POSITION (QUARTER-GRID) mode for room '{active_room['display_name']}'")


    # Generate source positions (4 sources)
    source_positions = generate_source_positions(active_room, name="v1")

    # Print and visualize grid for each source position
    print(f"\n{'='*80}")
    print(f"Visualizing receiver grids for {len(source_positions)} source positions")
    print(f"Receiver grid: {len(receiver_positions)} positions")
    print(f"{'='*80}\n")

    for idx, source_pos in enumerate(source_positions):
        source_x, source_y, source_z, source_name = source_pos
        print(f"\n{'='*80}")
        print(f"Source {idx + 1}/{len(source_positions)}: {source_name}")
        print(f"Position: ({source_x:.2f}, {source_y:.2f}, {source_z:.2f})")
        print(f"{'='*80}")
        
        # Create a temporary room dict with current source position for visualization
        room_with_source = active_room.copy()
        room_with_source['source x'] = source_x
        room_with_source['source y'] = source_y
        room_with_source['source z'] = source_z
        
        # Print receiver grid for this source position
        print_receiver_grid(
            receiver_positions, 
            room_with_source, 
            source_position=(source_x, source_y, source_z), save=True,
            source_name=source_name
        )

    # Calculate and print paths for all source-receiver combinations
    # Calculate and store phantom path stats
    print(f"\n{'='*80}")
    print(f"Calculating Phantom Path Ratios (Pruned/InValid Paths) for O2 and O3")
    print(f"{'='*80}\n")
    
    # Structure: results[order][receiver_idx] = { 'coords': (x,y), 'ratios': {source_name: ratio} }
    results = {
        1: {},
        2: {}
    }
    
    from sdn_path_calculator import ISMCalculator

    for receiver_idx, pos in enumerate(receiver_positions):
        rx, ry = pos[:2]
        rec_key = f"R{receiver_idx+1}"
        
        # Initialize rows for this receiver
        for order in [1,2]:
            results[order][receiver_idx] = {
                'coords': (rx, ry),
                'ratios': {} #ratios of
            }

        for source_idx, source_pos in enumerate(source_positions):
            source_x, source_y, source_z, source_name = source_pos
            
            # Setup room geometry
            room = geometry.Room(active_room['width'], active_room['depth'], active_room['height'])
            room.set_source(source_x, source_y, source_z, signal=None, Fs=44100)
            room.set_microphone(rx, ry, active_room['mic z'])
            
            # Create path tracker and calculator
            tracker = path_tracker.PathTracker()
            sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
            sdn_calc.set_path_tracker(tracker)
            
            ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
            ism_calc.set_path_tracker(tracker)

            # Analyze for
            for max_order in [1,2]:
                sdn_calc.calculate_paths_up_to_order(max_order)
                ism_calc.calculate_paths_up_to_order(max_order)
                
                sdn_paths_all = []
                ism_paths_valid = []
                
                # Collect paths for specific orders 1..max_order
                for o in range(1, max_order + 1):
                    sdn_paths_all.extend(tracker.get_paths_by_order(o, 'SDN'))
                    ism_paths_valid.extend([p for p in tracker.get_paths_by_order(o, 'ISM') if p.is_valid])
                
                # Count Analysis
                total_sdn = len(sdn_paths_all)
                total_ism = len(ism_paths_valid)
                phantom_count = total_sdn - total_ism
                count_ratio = (phantom_count / total_sdn * 100) if total_sdn > 0 else 0
                
                # Energy Analysis (1/distance weighting)
                sdn_energy = sum(1.0/p.distance for p in sdn_paths_all)
                ism_energy = sum(1.0/p.distance for p in ism_paths_valid)
                phantom_energy = sdn_energy - ism_energy
                energy_ratio = (phantom_energy / sdn_energy * 100) if sdn_energy > 0 else 0
                
                if 'counts' not in results[max_order][receiver_idx]: results[max_order][receiver_idx]['counts'] = {}
                if 'energy' not in results[max_order][receiver_idx]: results[max_order][receiver_idx]['energy'] = {}
                
                results[max_order][receiver_idx]['counts'][source_name] = count_ratio
                results[max_order][receiver_idx]['energy'][source_name] = energy_ratio
                
                print(f"Processed: Order {max_order} | {source_name} -> {rec_key}", end='\r')

    print("\nCalculation complete. Generating tables...")

    # Define Source Names for Columns
    source_names = [s[3] for s in source_positions]
    
    # Helper to print table
    def print_phantom_table(order, metric_type='counts'):
        title = "Phantom Path COUNTS (Pruned %)" if metric_type == 'counts' else "Phantom Path ENERGY (Pruned %)"
        print(f"\n{title} - Up to Order {order}")
        print("-" * 100)
        
        # Header
        header = f"{'Receiver':<10} | {'Coords (x, y)':<15} ||"
        for name in source_names:
            header += f" {name[:12]:<12} |"
        print(header)
        print("-" * 100)
        
        # Initialize column sums
        col_sums = {name: 0.0 for name in source_names}
        row_count = 0
        
        # Rows
        for rx_idx in sorted(results[order].keys()):
            data = results[order][rx_idx]
            rx, ry = data['coords']
            row_str = f"R{rx_idx+1:<9} | ({rx:<4.2f}, {ry:<4.2f})   ||"
            
            for src_name in source_names:
                ratio = data[metric_type].get(src_name, 0.0)
                col_sums[src_name] += ratio
                row_str += f" {ratio:>10.1f}% |"
            print(row_str)
            row_count += 1
            
        print("-" * 100)
        
        # Mean Row
        if row_count > 0:
            mean_str = f"{'MEAN':<10} | {'-':<15} ||"
            for src_name in source_names:
                avg = col_sums[src_name] / row_count
                mean_str += f" {avg:>10.1f}% |"
            print(mean_str)
            print("-" * 100)

    # Print Tables
    # Order 1 Count (Verification that it is 0)
    print_phantom_table(1, 'counts')
    
    # Order 2 Count (To compare with previous results ~25-30%)
    print_phantom_table(2, 'counts')
    
    # Order 2 Energy (Weighted view ~14-15%)
    print_phantom_table(2, 'energy')

