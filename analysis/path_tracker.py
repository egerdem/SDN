from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from .spatial_analysis import generate_receiver_grid_old, generate_source_positions, print_receiver_grid

@dataclass
class Path:
    nodes: List[str]  # List of node identifiers: ['s', 'f', 'w', 'm'] etc.
    distance: float   # Total path distance (from image source method)
    method: str      # 'SDN' or 'ISM'
    order: int       # Reflection order (number of bounces)
    is_valid: bool = True  # Add validity flag
    segment_distances: List[float] = None  # Individual segment distances
    
    # Wall name mapping for compact display
    _wall_abbrev = {
        'ceiling': 'c',
        'floor': 'f',
        'north': 'n',
        'south': 's',
        'east': 'e',
        'west': 'w'
    }
    
    def __str__(self):
        """Pretty print the path"""
        # Convert wall names to abbreviations
        path_nodes = [self._wall_abbrev.get(node, node) for node in self.nodes]
        path_str = ' → '.join(path_nodes)
        segments_str = ""
        if self.segment_distances:
            segments_str = f" ({' + '.join(f'{d:.2f}' for d in self.segment_distances)})"
        return f"{path_str:<10} Len: {self.distance:.2f}m{segments_str}"
    
    @property
    def total_segment_distance(self) -> float:
        """Calculate total distance by summing segments"""
        if self.segment_distances:
            return sum(self.segment_distances)
        return 0.0
    
    @property
    def distance_difference(self) -> float:
        """Calculate difference between total and sum of segments"""
        if self.segment_distances:
            return abs(self.distance - self.total_segment_distance)
        return 0.0

class PathTracker:
    def __init__(self):
        # Store paths by order for both methods
        self.paths: Dict[str, Dict[int, List[Path]]] = {
            'SDN': defaultdict(list),
            'ISM': defaultdict(list)
        }
        self.speed_of_sound = 343.0  # m/s
        
    def add_path(self, nodes: List[str], distance: float, method: str, 
                 is_valid: bool = True, segment_distances: List[float] = None):
        """Add a new path with its distance."""
        order = len(nodes) - 2  # Subtract source and mic
        path = Path(nodes=nodes, 
                    distance=distance, 
                    method=method, 
                    order=order, 
                    is_valid=is_valid,
                    segment_distances=segment_distances)
        self.paths[method][order].append(path)
    
    def get_paths_by_order(self, order: int, method: str) -> List[Path]:
        """Get all paths of a specific order for given method."""
        return self.paths[method][order]
    
    def print_all_paths_sorted(self):
        """Print all paths sorted by distance, regardless of order"""
        for method in ['SDN', 'ISM']:
            print(f"\n{method} Paths (Sorted by Distance):")
            print("=" * 50)
            
            # Collect all paths for this method
            all_paths = []
            for order_paths in self.paths[method].values():
                all_paths.extend(order_paths)
            
            # Sort paths by distance
            sorted_paths = sorted(all_paths, key=lambda p: p.distance)
            
            # Print paths with order information
            for path in sorted_paths:
                print(f"Order {path.order}: {str(path)}")
    
    def print_all_paths(self):
        """Print all paths grouped by method and order"""
        for method in ['SDN', 'ISM']:
            print(f"\n{method} Paths:")
            print("=" * 50)
            for order in sorted(self.paths[method].keys()):
                print(f"\nOrder {order} reflections:")
                print("-" * 40)
                paths = self.paths[method][order]
                for path in paths:
                    print(str(path)) 
    
    def print_path_comparison(self):
        """Print SDN and ISM paths side by side for comparison."""
        print("\nPath Comparison (SDN vs ISM):")
        room_name = room_parameters.get('display_name')
        rp = room_parameters
        src = rp["source x"], rp["source y"], rp["source z"]
        mic = rp["mic x"], rp["mic y"], rp["mic z"]
        room_info = f"**{room_name}: {rp['width']}×{rp['depth']}×{rp['height']}m " \
                    f"SOURCE: {src[0]:.2f}m, {src[1]:.2f}m, {src[2]:.2f}m " \
                    f"MICROPHONE: {mic[0]:.2f}m, {mic[1]:.2f}m, {mic[2]:.2f}m  **"
        print(f"{room_info}")
        print("=" * 80)
        print(f"{'SDN Path':<40} | {'ISM Path':<40}")
        print("-" * 80)
        
        # Group paths by their wall sequence (ignoring 's' and 'm')
        sdn_paths_by_sequence = {}
        ism_paths_by_sequence = {}
        
        # Collect paths and find max order
        max_order = 0
        for method, paths_by_order in self.paths.items():
            for order, order_paths in paths_by_order.items():
                max_order = max(max_order, order)
                for path in order_paths:
                    # Get wall sequence (exclude 's' and 'm')
                    wall_sequence = tuple(path.nodes[1:-1])
                    if method == 'SDN':
                        sdn_paths_by_sequence[wall_sequence] = path
                    else:
                        ism_paths_by_sequence[wall_sequence] = path
        
        # Find all unique sequences
        all_sequences = set(list(sdn_paths_by_sequence.keys()) + 
                           list(ism_paths_by_sequence.keys()))
        
        # If no paths found, print message and return
        if not all_sequences:
            print("No paths found.")
            return
        
        # Print paths by order
        for order in range(max_order + 1):
            print(f"\nOrder {order} reflections:")
            print("-" * 80)
            
            order_sequences = [seq for seq in all_sequences if len(seq) == order]
            
            for seq in sorted(order_sequences):
                sdn_path = sdn_paths_by_sequence.get(seq, None)
                ism_path = ism_paths_by_sequence.get(seq, None)
                
                # Include validity in the path strings
                sdn_str = str(sdn_path) if sdn_path else " "
                ism_str = str(ism_path) if ism_path else " "
                
                # Add difference in distances and validity info
                if sdn_path and ism_path:
                    diff = abs(sdn_path.distance - ism_path.distance)
                    validity_str = ""
                    if not sdn_path.is_valid or not ism_path.is_valid:
                        validity_str = "(INVALID)"
                    print(f"{sdn_str}       |      {ism_str}        (Δ = {diff:.2f}m) {validity_str}")
                else:
                    validity_str = "(INVALID)" if (sdn_path and not sdn_path.is_valid) or (ism_path and not ism_path.is_valid) else ""
                    print(f"{sdn_str}        |     {ism_str}        {validity_str}")

    def get_latest_arrival_time_by_order(self, method: str = 'ISM') -> Dict[int, float]:
        """Get the latest arrival time for each order considering only valid paths.
        
        Args:
            method: 'SDN' or 'ISM'
            
        Returns:
            Dictionary mapping order to latest arrival time in seconds
        """
        latest_times = {}
        for order, paths in self.paths[method].items():
            # Filter valid paths and get their distances
            valid_distances = [p.distance for p in paths if p.is_valid]
            if valid_distances:
                # Convert maximum distance to time using speed of sound
                latest_times[order] = max(valid_distances) / self.speed_of_sound
        return latest_times

    def count_valid_paths_up_to_order(self, max_order: int, method: str = 'ISM') -> int:
        """Count number of valid paths up to and including the specified order.
        
        Args:
            max_order: Maximum reflection order to include
            method: 'SDN' or 'ISM'
            
        Returns:
            Total number of valid paths from order 0 to max_order
        """
        valid_count = 0
        for order in range(max_order + 1):
            valid_count += sum(1 for path in self.paths[method][order] if path.is_valid)
        return valid_count

    def print_valid_paths_count(self, max_order: int):
        """Print the count of valid paths up to specified order for ISM."""
        valid_count = self.count_valid_paths_up_to_order(max_order, 'ISM')
        print(f"\nNumber of valid ISM paths up to order {max_order}: {valid_count}")
        
        # Print breakdown by order
        print("\nBreakdown by order:")
        for order in range(max_order + 1):
            count = sum(1 for path in self.paths['ISM'][order] if path.is_valid)
            print(f"Order {order}: {count} valid paths")

if __name__ == "__main__":
    from .sdn_path_calculator import SDNCalculator, ISMCalculator, PathCalculator
    import geometry
    from . import plot_room as pp
    import matplotlib.pyplot as plt

    plot_arrival_times_per_order = False

    room_aes = {'width': 9, 'depth': 7, 'height': 4,
                'source x': 4.5, 'source y': 3.5, 'source z': 2,
                'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                'absorption': 0.2,
                }

    room_journal = {'width': 3.2, 'depth': 4, 'height': 2.7,
                    'source x': 2, 'source y': 3., 'source z': 2,
                    'mic x': 1, 'mic y': 1, 'mic z': 1.5,
                    'absorption': 0.1,
                    }

    room_aes_outliar = {
        'display_name': 'AES Room',
        'width': 9, 'depth': 7, 'height': 4,
        'source x': 4.5, 'source y': 6, 'source z': 2,
        'mic x':0.5, 'mic y': 0.5, 'mic z': 1.5,
        'absorption': 0.2,
    }

    room_parameters = room_aes_outliar
    active_room = room_parameters
    room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])

    receiver_positions = generate_receiver_grid_old(active_room['width'] / 2, active_room['depth'] / 2, wall_margin=0.5,
                                                    center_margin=0.5,
                                                    n_points=16)
    source_pos = room_parameters['source x'], room_parameters['source y'], room_parameters['source z']

    print_receiver_grid(receiver_positions, room_parameters)

    for i, (rx, ry) in enumerate(receiver_positions):
        print(f"    ... receiver {i + 1}/{len(receiver_positions)} at ({rx:.2f}, {ry:.2f})")

        room_parameters.update({
            'mic x': rx,
            'mic y': ry,
            'source x': source_pos[0],
            'source y': source_pos[1],
            'source z': source_pos[2],
        })


        room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
        room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                        signal="will be replaced", Fs=44100)


        # Only Path Length Analysis, No RIR Calculation
        # Create shared path tracker and calculate paths
        path_tracker = PathTracker()
        sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
        ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
        sdn_calc.set_path_tracker(path_tracker)
        ism_calc.set_path_tracker(path_tracker)
        N = 1
        # Compare paths and analyze invalid ISM paths
        PathCalculator.compare_paths(sdn_calc, ism_calc,
                                     max_order=N, print_comparison=True)  # Increased to 5

        # analyze_paths() returns a list of invalid paths (only for ISM calculator)
        invalid_paths = ism_calc.analyze_paths(max_order=N)  # Increased to 5

        # Visualize example ISM paths
            # example_paths = [
            #     ['s', 'east', 'west', 'm'],
                # ['s', 'west', 'm'],
                # ['s', 'west', 'east', 'north', 'm']
            # ]

        # for path in example_paths:
        #     pp.plot_ism_path(room, ism_calc, path)
        #     plt.show()

        TOA_order_SDN = path_tracker.get_latest_arrival_time_by_order('SDN')
        TOA_order_ISM = path_tracker.get_latest_arrival_time_by_order('ISM')

        # Print arrival times
        print("\nLatest Arrival Times by Order:")
        print("=" * 50)
        print(f"{'Order':>5} {'SDN (s)':>10} {'ISM (s)':>10}")
        print("-" * 50)

        for order in range(N+1):
            sdn_time = f"{TOA_order_SDN.get(order, '-'):>10.5f}" if order in TOA_order_SDN else " " * 10
            ism_time = f"{TOA_order_ISM.get(order, '-'):>10.5f}" if order in TOA_order_ISM else " " * 10
            print(f"{order:>5} {sdn_time} {ism_time}")

        path_tracker.print_valid_paths_count(N)

        if plot_arrival_times_per_order:
            # Create scatter plot of arrival times vs order
            plt.figure(figsize=(10, 6))

            # Plot SDN points
            sdn_orders = []
            sdn_times = []
            for order in range(N+1):  # 0 to 4
                if order in TOA_order_SDN:
                    sdn_orders.append(order)
                    sdn_times.append(TOA_order_SDN[order])
            plt.scatter(sdn_times, sdn_orders, label='SDN', marker='o', s=100, alpha=0.6)

            # Plot ISM points
            ism_orders = []
            ism_times = []
            for order in range(1, N+1):  # 1 to 4 (ISM doesn't have 0th order)
                if order in TOA_order_ISM:
                    ism_orders.append(order)
                    ism_times.append(TOA_order_ISM[order])
            # Add 0th order from SDN to ISM (they're the same)
            if 0 in TOA_order_SDN:
                ism_orders.insert(0, 0)
                ism_times.insert(0, TOA_order_SDN[0])
            plt.scatter(ism_times, ism_orders, label='ISM', marker='x', s=100, alpha=0.6)

            plt.xlabel('Arrival Time (s)')
            plt.ylabel('Reflection Order')
            plt.title('Latest Arrival Times per Reflection Order')
            plt.grid(True)
            plt.legend()
            plt.show(block=False)

        paths = path_tracker.paths["SDN"]
        # Flatten and sort all paths by distance
        sorted_paths = sorted(
            (path for order_paths in paths.values() for path in order_paths),
            key=lambda p: p.distance
        )


# source_pos_new = [(1.0, 1.0, 6, 'Upper_Right_Source'),
#                   (2.0, 1.0, 7, "Center_SourceV2")]