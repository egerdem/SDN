"""
Visualize receiver grids for multiple source positions.

This script generates and displays receiver grids for each source position,
showing the spatial setup without running any experiments.
Also calculates and prints paths up to order 1 for all source-receiver combinations.
"""

from spatial_analysis import (
    print_receiver_grid,
    generate_receiver_grid_old,
    generate_source_positions, generate_full_receiver_grid
)
import geometry
from sdn_path_calculator import SDNCalculator
import path_tracker

if __name__ == "__main__":
    # Room parameters
    room_aes = {
        'display_name': 'AES Room',
        'width': 9, 'depth': 7, 'height': 4,
        'source x': 4.5, 'source y': 3.5, 'source z': 2,
        'mic z': 1.5,
        'absorption': 0.2,
    }

    # Generate receiver grid with 16 positions
    # Note: Using same pattern as spatial_analysis.py main block
    receiver_positions = generate_receiver_grid_old(
        room_aes['width'] / 2, 
        room_aes['depth'] / 2, 
        n_points=16,
        margin=0.5
    )

    receiver_positions = generate_full_receiver_grid(
    room_width=room_aes['width'],
    room_depth=room_aes['depth'],
    height=room_aes['mic z'],
    n_x=4,
    n_y=4,
    margin=0.5
)

    # Generate source positions (4 sources)
    source_positions = generate_source_positions(room_aes, name="v1")

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
        room_with_source = room_aes.copy()
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
    print(f"\n{'='*80}")
    print(f"Calculating SDN paths up to order 1 for all {len(source_positions)} sources × {len(receiver_positions)} receivers")
    print(f"Comparing SDN first-order paths with each other to find equal paths")
    print(f"Printing node positions when close (< 1m)")
    print(f"{'='*80}\n")

    for source_idx, source_pos in enumerate(source_positions):
        source_x, source_y, source_z, source_name = source_pos
        
        for receiver_idx, (rx, ry) in enumerate(receiver_positions):
            
            # Setup room geometry
            room = geometry.Room(room_aes['width'], room_aes['depth'], room_aes['height'])
            room.set_source(source_x, source_y, source_z, signal=None, Fs=44100)
            room.set_microphone(rx, ry, room_aes['mic z'])
            
            # Create path tracker and calculator
            tracker = path_tracker.PathTracker()
            
            # Calculate SDN paths only
            sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
            sdn_calc.set_path_tracker(tracker)
            sdn_calc.calculate_paths_up_to_order(max_order=1)
            
            # Get first order SDN paths
            sdn_paths_1 = tracker.get_paths_by_order(1, 'SDN')
            
            # Compare SDN paths with each other to find equal paths
            equal_paths = []
            close_nodes = []
            
            for i, path1 in enumerate(sdn_paths_1):
                for j, path2 in enumerate(sdn_paths_1):
                    if i < j:  # Avoid comparing path with itself and duplicate comparisons
                        # Extract wall sequence (remove 's' and 'm')
                        walls1 = path1.nodes[1:-1]  # Remove 's' at start and 'm' at end
                        walls2 = path2.nodes[1:-1]
                        
                        # Check if paths are equal (same wall sequence)
                        if walls1 == walls2:
                            equal_paths.append((path1, path2))
                        
                        # Check node positions for all paths (even if not equal)
                        wall_id1 = path1.nodes[1]
                        wall_id2 = path2.nodes[1]
                        
                        node_pos1 = room.walls[wall_id1].node_positions
                        node_pos2 = room.walls[wall_id2].node_positions
                        
                        node_distance = node_pos1.getDistance(node_pos2)
                        
                        # If nodes are close (< 1m), record it
                        if node_distance < 1.0:
                            close_nodes.append((path1, path2, wall_id1, wall_id2, node_pos1, node_pos2, node_distance))
            
            # Print results
            if equal_paths or close_nodes:
                print(f"\n{'='*80}")
                print(f"Source {source_idx + 1}/{len(source_positions)}: {source_name} "
                      f"({source_x:.2f}, {source_y:.2f}, {source_z:.2f})")
                print(f"Receiver {receiver_idx + 1}/{len(receiver_positions)}: ({rx:.2f}, {ry:.2f}, {room_aes['mic z']:.2f})")
                
                # Print equal paths
                if equal_paths:
                    print(f"\nFound {len(equal_paths)} equal first-order path pair(s):")
                    print("-" * 60)
                    for path1, path2 in equal_paths:
                        wall_id = path1.nodes[1]
                        print(f"\nEqual paths:")
                        print(f"  Path 1: {path1}")
                        print(f"  Path 2: {path2}")
                        print(f"  Wall: {wall_id}")
                
                # Print close nodes
                if close_nodes:
                    print(f"\nFound {len(close_nodes)} pair(s) with close node positions (< 1m):")
                    print("-" * 60)
                    for path1, path2, wall_id1, wall_id2, node_pos1, node_pos2, node_distance in close_nodes:
                        print(f"\nClose nodes:")
                        print(f"  Path 1: {path1} (wall: {wall_id1})")
                        print(f"  Path 2: {path2} (wall: {wall_id2})")
                        print(f"  Node 1 position: ({node_pos1.x:.3f}, {node_pos1.y:.3f}, {node_pos1.z:.3f})")
                        print(f"  Node 2 position: ({node_pos2.x:.3f}, {node_pos2.y:.3f}, {node_pos2.z:.3f})")
                        print(f"  Distance: {node_distance:.3f} m")

