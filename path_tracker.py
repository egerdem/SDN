from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

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


if __name__ == "__main__":
    from sdn_path_calculator import SDNCalculator, ISMCalculator, PathCalculator
    import geometry
    import plot_room as pp
    import matplotlib.pyplot as plt

    room_aes = {'width': 9, 'depth': 7, 'height': 4,
                'source x': 4.5, 'source y': 3.5, 'source z': 2,
                'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                'absorption': 0.2,
                'air': {'humidity': 50,
                        'temperature': 20,
                        'pressure': 100},
                }

    room_journal = {'width': 3.2, 'depth': 4, 'height': 2.7,
                    'source x': 2, 'source y': 3., 'source z': 2,
                    'mic x': 1, 'mic y': 1, 'mic z': 1.5,
                    'absorption': 0.1,
                    }

    room_parameters = room_aes

    room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
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

    # Compare paths and analyze invalid ISM paths
    PathCalculator.compare_paths(sdn_calc, ism_calc,
                                 max_order=2)  # compare_paths() prints the comparison table but doesn't return anything

    # analyze_paths() returns a list of invalid paths (only for ISM calculator)
    # Each path is a list of node labels ['s', 'wall1', 'wall2', ..., 'm']
    invalid_paths = ism_calc.analyze_paths(max_order=2)

    # Visualize example ISM paths
    example_paths = [
        ['s', 'east', 'west', 'm'],
        # ['s', 'west', 'm'],
        # ['s', 'west', 'east', 'north', 'm']
    ]

    for path in example_paths:
        pp.plot_ism_path(room, ism_calc, path)
        plt.show()