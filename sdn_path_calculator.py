from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional
import geometry
from geometry import Point
import path_tracker
from collections import defaultdict

class PathCalculator(ABC):
    """Abstract base class for path calculation methods"""
    def __init__(self, walls: Dict[str, 'Wall'], source_pos: Point, mic_pos: Point):
        self.walls = walls
        self.source_pos = source_pos
        self.mic_pos = mic_pos
        self.path_tracker = path_tracker.PathTracker()
    
    @abstractmethod
    def calculate_paths_up_to_order(self, max_order: int):
        pass

class SDNCalculator(PathCalculator):
    """SDN specific calculator with fixed nodes"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fixed nodes for SDN
        self.node_labels = ['s', 'm'] + list(self.walls.keys())
        self.node_to_idx = {label: idx for idx, label in enumerate(self.node_labels)}
        n_nodes = len(self.node_labels)
        self.node_distances = np.zeros((n_nodes, n_nodes))
        self.node_distances = self.calculate_node_distances()
    
    def calculate_node_distances(self):
        """Calculate distances between all nodes using numpy array."""
        # Get all node positions in same order as node_labels
        positions = np.vstack([
            self.source_pos.to_array(),
            self.mic_pos.to_array(),
            *[self.walls[wall_id].node_positions.to_array() for wall_id in self.walls]
        ])
        
        # Initialize fresh distance matrix
        n_nodes = len(self.node_labels)
        distances = np.zeros((n_nodes, n_nodes))
        
        # Calculate all pairwise distances at once
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    distances[i,j] = np.linalg.norm(positions[i] - positions[j])
        
        return distances
    
    def get_distance(self, node1: str, node2: str) -> float:
        """Get distance between two nodes by their labels."""
        idx1 = self.node_to_idx[node1]
        idx2 = self.node_to_idx[node2]
        return self.node_distances[idx1, idx2]
    
    def calculate_paths_up_to_order(self, max_order: int):
        """Calculate all SDN paths up to specified order."""
        # First order paths (direct connections to each wall then to mic)
        for wall_id in self.walls:
            path = ['s', wall_id, 'm']
            distance = (self.get_distance('s', wall_id) + 
                       self.get_distance(wall_id, 'm'))
            self.path_tracker.add_path(path, distance, 'SDN')
        
        # Higher order paths
        for order in range(2, max_order + 1):
            self._calculate_paths_of_order(order)
    
    def _calculate_paths_of_order(self, order: int):
        """Calculate paths of a specific order."""
        def generate_paths(current_path: List[str], remaining_bounces: int):
            if remaining_bounces == 0:
                # Add microphone and calculate total distance
                distance = 0
                for i in range(len(current_path)-1):
                    distance += self.get_distance(current_path[i], current_path[i+1])
                distance += self.get_distance(current_path[-1], 'm')
                
                self.path_tracker.add_path(current_path + ['m'], distance, 'SDN')
                return
            
            # Try each wall as next bounce
            for wall_id in self.walls:
                if not current_path or current_path[-1] != wall_id:  # Avoid consecutive same-wall bounces
                    generate_paths(current_path + [wall_id], remaining_bounces - 1)
        
        # Start paths from source
        generate_paths(['s'], order)

class ISMCalculator(PathCalculator):
    """ISM specific calculator"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_sources = {}  # Dict[tuple[str], Point]  # path -> image source position
        self.node_positions = {}  # Dict[tuple[str], List[Point]]  # path -> list of intersection points
    
    def calculate_image_source(self, path: List[str]) -> Point:
        """Calculate position of image source for given reflection path."""
        current_source = self.source_pos
        
        for wall_id in path[1:]:  # Skip 's' in path
            wall = self.walls[wall_id]
            img_source_calc = geometry.ImageSource({wall_id: wall}, current_source, self.mic_pos)
            current_source = img_source_calc.IS_1st_order(wall)
            
        return current_source
    
    def calculate_intersection_points(self, path: List[str], final_image_source: Point) -> List[Point]:
        """Calculate intersection points for a path working backwards from mic."""
        points = []
        walls_reversed = list(reversed(path[1:]))  # Skip 's', reverse wall order
        
        # Start with mic -> final image source for last reflection
        current_point = self.mic_pos
        current_image = final_image_source
        
        for i, wall_id in enumerate(walls_reversed):
            wall = self.walls[wall_id]
            img_source_calc = geometry.ImageSource({wall_id: wall}, current_image, current_point)
            intersection = img_source_calc.findIntersectionPoint(current_point, current_image, wall)
            points.insert(0, intersection)  # Insert at beginning to maintain path order
            
            # For next iteration, use previous reflection point and corresponding image source
            current_point = intersection
            if i < len(walls_reversed) - 1:  # If not the last (first) reflection
                prev_path = ['s'] + path[1:-(i+1)]  # Path up to next wall
                current_image = self.calculate_image_source(prev_path)
        
        return points
    
    def calculate_paths_up_to_order(self, max_order: int):
        """Generate all possible ISM paths up to given order."""
        # First order paths
        for wall_id in self.walls:
            path = ['s', wall_id]
            image_source = self.calculate_image_source(path)
            self.image_sources[tuple(path)] = image_source
            intersections = self.calculate_intersection_points(path, image_source)
            self.node_positions[tuple(path)] = intersections
            distance = image_source.getDistance(self.mic_pos)
            self.path_tracker.add_path(path + ['m'], distance, 'ISM')
        
        # Higher order paths
        for order in range(2, max_order + 1):
            def generate_paths(current_path: List[str], remaining_bounces: int):
                if remaining_bounces == 0:
                    image_source = self.calculate_image_source(current_path)
                    self.image_sources[tuple(current_path)] = image_source
                    intersections = self.calculate_intersection_points(current_path, image_source)
                    self.node_positions[tuple(current_path)] = intersections
                    distance = image_source.getDistance(self.mic_pos)
                    self.path_tracker.add_path(current_path + ['m'], distance, 'ISM')
                    return
                
                for wall_id in self.walls:
                    if not current_path or current_path[-1] != wall_id:  # Avoid consecutive same-wall bounces
                        generate_paths(current_path + [wall_id], remaining_bounces - 1)
            
            generate_paths(['s'], order)
