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