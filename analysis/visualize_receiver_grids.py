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
    print_source_grid,
    generate_receiver_grid_old,
    generate_source_positions, generate_full_receiver_grid, generate_fixed_source_grid,
    generate_fixed_source_grid_simple, merge_source_lists
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

    # Feature Flags
    PATH_CALC = False  # Enable/disable path calculation analysis
    HSRC_Contour = True  # Enable H_src contour plot visualization
    PLOT_SOURCE_LOCATIONS = True  # Plot source positions on H_src contour map

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
            margin=0.5, z=active_room['mic z']
        )
        print(f"Running in MULTI-POSITION (QUARTER-GRID) mode for room '{active_room['display_name']}'")

# Generate source positions (v3 includes corner source)
    # source_positions = generate_source_positions(active_room, name="v3")
    #source_positions_9src = generate_fixed_source_grid(active_room, padding=0.5, n_x=3, n_y=3, z_mode="fixed_1p5",
    #include_corners=True, corner_offset=1)

    #source_positions_4src = generate_fixed_source_grid_simple(active_room, padding=2, n_x=2, n_y=2, z_mode="fixed_1p5") #deneydeki
    #source_positions = merge_source_lists(source_positions_9src, source_positions_4src)
    
    # Generate source positions - using diagonal grid for H_src analysis
    source_positions = generate_fixed_source_grid(
        active_room,
        padding=0.5,
        n_x=15,
        n_y=15,
        z_mode="fixed_1p5",
        include_corners=True,
        corner_offset=0.5,
        diagonals="3d"
    )

    # Print and visualize grid for each source position
    print(f"\n{'='*80}")
    print(f"Visualizing receiver grids for {len(source_positions)} source positions")
    print(f"Receiver grid: {len(receiver_positions)} positions")
    print(f"{'='*80}\n")

    # Use combined visualization if there are more than 5 sources
    if len(source_positions) >= 4:
        print(f"Using combined source-receiver grid visualization (>={4} sources)")
        print_source_grid(
            source_positions, 
            receiver_positions, 
            active_room, 
            save=False,
            grid_name="fixed_grid"
        )
    else:
        # Use individual plots for each source (original behavior)
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
                source_position=(source_x, source_y, source_z), save=False,
                source_name=source_name
            )

    # Calculate and print paths for all source-receiver combinations
    # Calculate and store phantom path stats
    if PATH_CALC:
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
        # print_phantom_table(1, 'counts')
        
        # Order 2 Count (To compare with previous results ~25-30%)
        # print_phantom_table(2, 'counts')
        
        # Order 2 Energy (Weighted view ~14-15%)
        # print_phantom_table(2, 'energy')


    # ============================================================================
    # NEW: Path Length Overlap Analysis for ALL Sources and ALL Receivers
    # ============================================================================
    if PATH_CALC:
        print(f"\n{'='*80}")
        print(f"PATH LENGTH OVERLAP ANALYSIS")
        print(f"Checking if 2nd order paths can be shorter than 1st order paths")
        print(f"{'='*80}\n")
        
        # Structure to store overlap data: [receiver_idx][source_name] = overlap_amount
        overlap_data = {}
        
        # Analyze for ALL sources
        for source_idx, source_pos in enumerate(source_positions):
            source_x, source_y, source_z, source_name = source_pos
            print(f"\nProcessing Source: {source_name} ({source_x:.2f}, {source_y:.2f}, {source_z:.2f})")
            
            # Analyze ALL 16 receiver positions
            for receiver_idx, pos in enumerate(receiver_positions):
                rx, ry = pos[:2]
                
                if receiver_idx not in overlap_data:
                    overlap_data[receiver_idx] = {'coords': (rx, ry), 'sources': {}}
                
                # Setup room geometry
                room = geometry.Room(active_room['width'], active_room['depth'], active_room['height'])
                room.set_source(source_x, source_y, source_z, signal=None, Fs=44100)
                room.set_microphone(rx, ry, active_room['mic z'])
                
                # Create path tracker and calculators
                tracker = path_tracker.PathTracker()
                sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
                ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
                sdn_calc.set_path_tracker(tracker)
                ism_calc.set_path_tracker(tracker)
                
                # Calculate paths up to order 2
                max_order = 2
                sdn_calc.calculate_paths_up_to_order(max_order)
                ism_calc.calculate_paths_up_to_order(max_order)
                
                # Get paths for each order
                order1_ism = [p for p in tracker.get_paths_by_order(1, 'ISM') if p.is_valid]
                order2_ism = [p for p in tracker.get_paths_by_order(2, 'ISM') if p.is_valid]
                
                # Collect lengths
                order1_lengths = [p.distance for p in order1_ism]
                order2_lengths = [p.distance for p in order2_ism]
                
                # Calculate overlap (max path difference when O2 < O1)
                overlap_amount = 0.0
                if order1_lengths and order2_lengths:
                    max_order1 = max(order1_lengths)
                    min_order2 = min(order2_lengths)
                    
                    if min_order2 < max_order1:
                        overlap_amount = max_order1 - min_order2
                
                overlap_data[receiver_idx]['sources'][source_name] = overlap_amount
                
                print(f"  R{receiver_idx+1:2d} @ ({rx:.2f}, {ry:.2f}): overlap = {overlap_amount:.3f}m", end='\r')
            
            print(f"  Completed {source_name}" + " " * 50)
        
        # Print detailed overlap information
        print(f"\n{'='*80}")
        print(f"RECEIVERS WITH OVERLAP (2nd order shorter than 1st order)")
        print(f"{'='*80}\n")
        
        for receiver_idx in sorted(overlap_data.keys()):
            rx, ry = overlap_data[receiver_idx]['coords']
            sources_with_overlap = {src: amt for src, amt in overlap_data[receiver_idx]['sources'].items() if amt > 0}
            
            if sources_with_overlap:
                print(f"R{receiver_idx+1:2d} @ ({rx:.2f}, {ry:.2f}):")
                for src_name, overlap_amt in sources_with_overlap.items():
                    print(f"  ⚠️  {src_name}: {overlap_amt:.3f}m overlap")
        
        # Print comprehensive table
        print(f"\n{'='*80}")
        print(f"OVERLAP TABLE: Max Path Difference (O1_max - O2_min) in meters")
        print(f"Positive values indicate 2nd order paths shorter than 1st order")
        print(f"{'='*80}\n")
        
        # Get source names for columns
        source_names = [s[3] for s in source_positions]
        
        # Print header
        header = f"{'Receiver':<12} | {'Coords (x,y)':<14} ||"
        for name in source_names:
            # Truncate long names for table width
            short_name = name[:12]
            header += f" {short_name:<12} |"
        print(header)
        print("-" * len(header))
        
        # Print rows for each receiver
        for receiver_idx in sorted(overlap_data.keys()):
            rx, ry = overlap_data[receiver_idx]['coords']
            row_str = f"R{receiver_idx+1:<11} | ({rx:>4.2f},{ry:>4.2f})  ||"
            
            for src_name in source_names:
                overlap_amt = overlap_data[receiver_idx]['sources'].get(src_name, 0.0)
                if overlap_amt > 0:
                    row_str += f" {overlap_amt:>10.3f}m |"
                else:
                    row_str += f" {'---':>12} |"
            
            print(row_str)
        
        print("-" * len(header))
        
        # Print column statistics (mean overlap per source)
        print(f"\n{'Mean':<12} | {'-':<14} ||", end='')
        for src_name in source_names:
            overlaps = [overlap_data[r]['sources'].get(src_name, 0.0) 
                       for r in overlap_data.keys()]
            # Only count receivers with overlap > 0 for mean
            nonzero_overlaps = [o for o in overlaps if o > 0]
            if nonzero_overlaps:
                mean_overlap = sum(nonzero_overlaps) / len(nonzero_overlaps)
                print(f" {mean_overlap:>10.3f}m |", end='')
            else:
                print(f" {'---':>12} |", end='')
        print()
        
        print(f"{'Count':<12} | {'-':<14} ||", end='')
        for src_name in source_names:
            count = sum(1 for r in overlap_data.keys() 
                       if overlap_data[r]['sources'].get(src_name, 0.0) > 0)
            print(f" {count:>10d}/16 |", end='')
        print()
        
        print("-" * len(header))
        print(f"\n{'='*80}")

    # ============================================================================
    # H_src Contour Plot Visualization
    # ============================================================================
    if HSRC_Contour:
        print(f"\n{'='*80}")
        print(f"H_src CONTOUR PLOT VISUALIZATION")
        print(f"Generating dense source grid with H_src harmonic mean values")
        print(f"{'='*80}\n")
        
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from deprecated.wall_proximity import harmonic_mean, get_wall_distances
        
        width = active_room['width']
        depth = active_room['depth']
        height = active_room['height']
        
        # Calculate characteristic length for normalization
        # Using cube root of volume as characteristic length
        char_length = (width * depth * height) ** (1/3)
        print(f"Room dimensions: W={width:.2f}m, D={depth:.2f}m, H={height:.2f}m")
        print(f"Characteristic length: {char_length:.3f}m\n")
        
        # Dense FULL 2D grid for the contour field (z = mid height).
        # NOTE: must NOT be diagonals="3d" — that returns collinear points and
        # Delaunay triangulation fails ("singular input data"). The diagonal
        # experiment sources are overlaid separately as markers below.
        contour_source_positions = generate_fixed_source_grid(
            active_room,
            padding=0.5,
            n_x=20,
            n_y=20,
            z_mode="mid",
            include_corners=True,
            corner_offset=0.5,
            diagonals=False,
        )

        
        # Extract coordinates and calculate H_src for each position
        src_x_list = []
        src_y_list = []
        h_src_list = []
        
        for idx, (sx, sy, sz) in enumerate(contour_source_positions):
            dists = get_wall_distances(sx, sy, sz, width, depth, height)
            h_src = harmonic_mean(dists)
            h_src_normalized = h_src / char_length  # Normalize by characteristic length
            
            src_x_list.append(sx)
            src_y_list.append(sy)
            h_src_list.append(h_src_normalized)
            
            if (idx + 1) % 50 == 0:
                print(f"Progress: {idx+1}/{len(contour_source_positions)} points", end='\r')
        
        print(f"\nH_src (normalized) range: [{min(h_src_list):.3f}, {max(h_src_list):.3f}]")
        
        # Create the contour plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create triangulation for irregular grid
        from matplotlib.tri import Triangulation
        triang = Triangulation(src_x_list, src_y_list)
        
        # Filled contour plot
        contour_filled = ax.tricontourf(triang, h_src_list, levels=20, cmap='RdYlGn_r', alpha=0.8)
        
        # Contour lines
        contour_lines = ax.tricontour(triang, h_src_list, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Colorbar (with shrink to prevent overflow)
        cbar = plt.colorbar(contour_filled, ax=ax, label='H_src / L_char (Normalized Harmonic Mean)', 
                           shrink=0.8, fraction=0.046, pad=0.04)
        
        # Draw room boundaries
        ax.plot([0, width, width, 0, 0], [0, 0, depth, depth, 0], 'k-', linewidth=2, label='Room Boundaries')
        
        # Plot the actual source positions used in the experiment (if flag is enabled)
        if PLOT_SOURCE_LOCATIONS and len(source_positions) > 0:
            src_x = [s[0] for s in source_positions]
            src_y = [s[1] for s in source_positions]
            ax.scatter(src_x, src_y, c='blue', s=100, marker='o', linewidths=2.5,
                      label='Source Positions (Diagonal Grid)', zorder=5)
            
            # Add coordinate labels for each source
            for sx, sy, sz in source_positions:
                ax.text(sx + 0.53, sy + 0.03, f'({sx:.1f},{sy:.1f},{sz:.1f})', 
                       fontsize=8, ha='center', va='top', color='black')
        
        # Labels and title
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        # ax.set_title(f'Normalized H_src Contour Map - Room {room_name[0]}',
        #             fontsize=14, fontweight='bold') # Lower values = closer to walls = harder scenario
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save the figure
        output_dir = os.path.join(project_root, 'results', 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'hsrc_contour_normalized_{room_name.replace(" ", "_")}_3ddiag.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nContour plot saved to: {output_path}")
        plt.show()



