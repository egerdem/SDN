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
        self.path_tracker = None  # Will be set via setter
    
    def set_path_tracker(self, tracker: path_tracker.PathTracker):
        """Set shared path tracker instance"""
        self.path_tracker = tracker
    
    @abstractmethod
    def calculate_paths_up_to_order(self, max_order: int):
        pass

class SDNCalculator(PathCalculator):
    """
    SDN calculator that uses fixed nodes for path approximation.
    All SDN paths are considered valid by definition.
    """
    def __init__(self, walls: Dict[str, 'Wall'], source_pos: Point, mic_pos: Point):
        super().__init__(walls, source_pos, mic_pos)
        # Initialize node structure
        self.node_labels = ['s', 'm'] + list(self.walls.keys())
        self.node_to_idx = {label: idx for idx, label in enumerate(self.node_labels)}
        self.node_distances = self.calculate_node_distances()
    
    def calculate_node_distances(self) -> np.ndarray:
        """Calculate distances between all nodes."""
        positions = np.vstack([
            self.source_pos.to_array(),
            self.mic_pos.to_array(),
            *[self.walls[wall_id].node_positions.to_array() for wall_id in self.walls]
        ])
        
        n_nodes = len(positions)
        distances = np.zeros((n_nodes, n_nodes))
        
        # Calculate all pairwise distances
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):  # Only calculate upper triangle
                distances[i,j] = distances[j,i] = np.linalg.norm(positions[i] - positions[j])
        
        return distances
    
    def calculate_paths_up_to_order(self, max_order: int):
        """Calculate all SDN paths up to specified order."""

        # Zeroth order paths: Direct sound
        self._calculate_zeroth_order_paths()

        # First order paths
        self._calculate_first_order_paths()
        
        # Higher order paths
        for order in range(2, max_order + 1):
            self._calculate_nth_order_paths(order)


    def _calculate_first_order_paths(self):
        """Calculate direct wall reflection paths."""
        for wall_id in self.walls:
            path = ['s', wall_id, 'm']
            # Calculate individual segments
            d1 = self._get_distance('s', wall_id)  # source to wall
            d2 = self._get_distance(wall_id, 'm')  # wall to mic
            total_distance = d1 + d2
            self.path_tracker.add_path(
                nodes=path,
                distance=total_distance,
                method='SDN',
                segment_distances=[d1, d2]
            )
    
    def _calculate_nth_order_paths(self, order: int):
        """Calculate paths with specified number of bounces."""
        def generate_paths(current_path: List[str], remaining_bounces: int):
            if remaining_bounces == 0:
                path = current_path + ['m']
                # Calculate segment distances
                segment_distances = [
                    self._get_distance(path[i], path[i+1]) 
                    for i in range(len(path)-1)
                ]
                total_distance = sum(segment_distances)
                self.path_tracker.add_path(
                    nodes=path,
                    distance=total_distance,
                    method='SDN',
                    segment_distances=segment_distances
                )
                return
            
            for wall_id in self.walls:
                if not current_path or current_path[-1] != wall_id:
                    generate_paths(current_path + [wall_id], remaining_bounces - 1)
        
        generate_paths(['s'], order)
    
    def _get_distance(self, node1: str, node2: str) -> float:
        """Get distance between two nodes by their labels."""
        idx1, idx2 = self.node_to_idx[node1], self.node_to_idx[node2]
        return self.node_distances[idx1, idx2]
    
    def _calculate_path_distance(self, path: List[str]) -> float:
        """Calculate total distance along a path."""
        return sum(self._get_distance(path[i], path[i+1]) 
                  for i in range(len(path)-1))
    
    def _calculate_zeroth_order_paths(self):
        """Calculate direct sound path from source to microphone."""
        path = ['s', 'm']
        # Calculate direct distance using the pre-computed distance matrix
        d0 = self._get_distance('s', 'm')  # source to mic
        
        self.path_tracker.add_path(
            nodes=path,
            distance=d0,
            method='SDN',
            segment_distances=[d0]
        )

class ISMCalculator(PathCalculator):
    """
    Image Source Method (ISM) calculator that finds exact reflection paths.
    
    The calculation flow is:
    1. Initialize empty storage for image sources and reflection points
    2. For each order N:
        a. Generate all possible N-bounce paths
        b. For each path:
            - Calculate final image source
            - Calculate reflection points backwards
            - Validate path geometry
            - Store valid paths with distances
    """
    def __init__(self, walls: Dict[str, 'Wall'], source_pos: Point, mic_pos: Point):
        super().__init__(walls, source_pos, mic_pos)
        # Storage for calculated results
        self.image_sources: Dict[tuple[str], Point] = {}  # path -> image source position
        self.node_positions: Dict[tuple[str], List[Point]] = {}  # path -> reflection points
    
    def calculate_paths_up_to_order(self, max_order: int):
        """Main entry point: calculate all paths up to given order."""
        # First order paths (direct wall reflections)
        self._calculate_first_order_paths()
        
        # Higher order paths (multiple bounces)
        for order in range(2, max_order + 1):
            self._calculate_nth_order_paths(order)
    
    def _calculate_first_order_paths(self):
        """Calculate and store all first-order reflection paths."""
        for wall_id in self.walls:
            path = ['s', wall_id]
            self._process_single_path(path)
    
    def _calculate_nth_order_paths(self, order: int):
        """Calculate all paths of specified order."""
        def generate_paths(current_path: List[str], remaining_bounces: int):
            if remaining_bounces == 0:
                self._process_single_path(current_path)
                return
            
            for wall_id in self.walls:
                if not current_path or current_path[-1] != wall_id:  # Avoid consecutive same-wall bounces
                    generate_paths(current_path + [wall_id], remaining_bounces - 1)
        
        generate_paths(['s'], order)
    
    def _process_single_path(self, path: List[str]):
        """Process a single reflection path: calculate, validate, and store."""
        # Calculate final image source
        image_source = self._calculate_image_source(path)
        self.image_sources[tuple(path)] = image_source
        
        # Calculate and validate reflection points
        reflection_points, is_valid = self._calculate_reflection_points(path, image_source)
        self.node_positions[tuple(path)] = reflection_points
        
        # Calculate total distance using image source method
        total_distance = image_source.getDistance(self.mic_pos)
        
        # Calculate segment distances
        points = [self.source_pos] + reflection_points + [self.mic_pos]
        segment_distances = [points[i].getDistance(points[i+1]) 
                            for i in range(len(points)-1)]
        
        # Store path with both distances
        self.path_tracker.add_path(
            nodes=path + ['m'],
            distance=total_distance,
            method='ISM',
            is_valid=is_valid,
            segment_distances=segment_distances
        )
    
    def _calculate_image_source(self, path: List[str]) -> Point:
        """Calculate position of final image source for given reflection path."""
        current_source = self.source_pos
        for wall_id in path[1:]:  # Skip 's' in path
            wall = self.walls[wall_id]
            img_source_calc = geometry.ImageSource({'wall': wall}, current_source, self.mic_pos)
            current_source = img_source_calc.get_first_order_image(wall)
        return current_source
    
    def _calculate_reflection_points(self, path: List[str], final_image_source: Point) -> tuple[List[Point], bool]:
        """Calculate and validate reflection points working backwards from mic."""
        points = []
        is_valid = True
        
        walls_reversed = list(reversed(path[1:]))  # Skip 's', reverse wall order
        current_point = self.mic_pos
        current_image = final_image_source
        
        for i, wall_id in enumerate(walls_reversed):
            wall = self.walls[wall_id]
            # Create ImageSource instance just for intersection calculation
            img_source_calc = geometry.ImageSource({'wall': wall}, current_image, current_point)
            intersection = img_source_calc._find_intersection_point(current_point, current_image)
            
            if not wall.is_point_within_bounds(intersection):
                is_valid = False
            
            points.insert(0, intersection)
            
            # Prepare for next iteration
            current_point = intersection
            if i < len(walls_reversed) - 1:
                prev_path = ['s'] + path[1:-(i+1)]
                current_image = self._calculate_image_source(prev_path)
        
        return points, is_valid
