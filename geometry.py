from __future__ import annotations  # This allows forward references in type hints
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5


class Room:
    """
    Class defining a room with some properties that can be controlled
    """

    def __init__(self, lx, ly, lz):
        self.label = "cuboid"
        self.nWalls = 6
        self.x = lx
        self.y = ly
        self.z = lz
        self.wallAttenuation = []  # this is a list
        self.wallFilters = dict()  # this is a dictionary
        self.angle_mappings = None  # Will be set after source and nodes are set up
        self.source = None
        self.micPos = None
        self.sdn_nodes_calculated = False  # New flag to track if nodes are calculated
        self._setup_walls()

    def _setup_walls(self):
        # Setting up walls (internal use only)
        # Wall order: south, north, west, east, ceiling, floor
        wall_order = ['south', 'north', 'west', 'east', 'ceiling', 'floor']
        self.walls = {
            'south': Wall(Point(0, 0, 0), Point(self.x, 0, 0), Point(0, 0, self.z)),
            'north': Wall(Point(0, self.y, 0), Point(0, self.y, self.z), Point(self.x, self.y, 0)),
            'west': Wall(Point(0, 0, 0), Point(0, self.y, 0), Point(0, 0, self.z)),
            'east': Wall(Point(self.x, 0, 0), Point(self.x, self.y, 0), Point(self.x, self.y, self.z)),
            'ceiling': Wall(Point(0, 0, self.z), Point(0, self.y, self.z), Point(self.x, self.y, self.z)),
            'floor': Wall(Point(0, 0, 0), Point(0, self.y, 0), Point(self.x, self.y, 0))
        }
        
        # Add wall indices based on the defined order
        for i, wall_id in enumerate(wall_order):
            if wall_id in self.walls:
                self.walls[wall_id].wall_index = i

    def set_source(self, sx, sy, sz, signal=None, Fs=44100):
        """Set source position and optionally its signal."""
        self.source = Source(sx, sy, sz, signal, Fs)
        self._try_calculate_sdn_nodes()  # Try to calculate nodes if mic is already set

    def set_microphone(self, mx, my, mz):
        """Set microphone position."""
        self.micPos = Point(mx, my, mz)
        self._try_calculate_sdn_nodes()  # Try to calculate nodes if source is already set

    def _try_calculate_sdn_nodes(self):
        """Try to calculate SDN nodes if both source and mic are set."""
        if self.source is not None and self.micPos is not None and not self.sdn_nodes_calculated:
            self._calculate_sdn_nodes()
            self.angle_mappings = build_angle_mappings(self)
            self.sdn_nodes_calculated = True

    def _calculate_sdn_nodes(self):
        """Calculate SDN node positions at first-order reflection points."""
        if self.source is None or self.micPos is None:
            raise ValueError("Both source and microphone positions must be set before calculating SDN nodes")
            
        for wall_label, wall in self.walls.items():
            # Calculate image source for "wall"
            img_source = ImageSource({wall_label: wall}, self.source.srcPos, self.micPos)
            # Find intersection point between IM-mic line segment and the wall
            wall.node_positions = img_source._find_intersection_point(img_source.imageSourcePos, img_source.micPos)


class Source:
    def __init__(self, sx, sy, sz, signal, Fs):
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.srcPos = Point(sx, sy, sz)
        self.signal = signal
        self.Fs = Fs
        # self.directivity = 1.0
        # self.heading = 0.0

    @staticmethod
    def generate_signal(label: str, num_samples: int, Fs: int = 44100):
        """Generate source signal based on label type.
        
        Args:
            label: Type of signal ('dirac' or 'gaussian')
            num_samples: Number of samples in the signal
            Fs: Sampling frequency (default: 44100)
            
        Returns:
            dict: Contains the signal array and label
        """
        if label == 'dirac':
            # Generate Dirac Impulse
            signal = np.array([1.0] + [0.0] * (num_samples - 1))
        elif label == 'gaussian':
            # Generate Gaussian pulse
            from analysis import frequency as ff
            signal = ff.gaussian_impulse(num_samples, num_gaussian_samples=30, std_dev=5, plot=False)
        else:
            raise ValueError('Invalid source label')

        return {'signal': signal,
                'label': label}


class ImageSource:
    def __init__(self, walls: Dict[str, Wall], srcPos: Point, micPos: Point):
        self.walls = walls
        self.srcPos = srcPos
        self.micPos = micPos
        self.imageSourcePos = None  # Will be set by get_first_order_image
        self._findImageSources()  # Calculate image sources during initialization

    def _findImageSources(self):
        for wall_label, wall in self.walls.items():
            self.imageSourcePos = self.get_first_order_image(wall)
            wall.IMS = self.imageSourcePos

    def get_first_order_image(self, wall):
        # find image source locations along the plane
        self.d = wall.plane_coeffs.d
        self.a = wall.plane_coeffs.a
        self.b = wall.plane_coeffs.b
        self.c = wall.plane_coeffs.c

        # Compute the distance from the point to the plane
        # Distance formula: (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2)
        norm = self.a ** 2 + self.b ** 2 + self.c ** 2
        dist_to_plane = (self.a * self.srcPos.x + self.b * self.srcPos.y + 
                        self.c * self.srcPos.z + self.d) / norm

        # Compute the reflection point
        self.ImageSource_Pos = Point(0.0, 0.0, 0.0)
        self.ImageSource_Pos.x = self.srcPos.x - 2 * dist_to_plane * self.a
        self.ImageSource_Pos.y = self.srcPos.y - 2 * dist_to_plane * self.b
        self.ImageSource_Pos.z = self.srcPos.z - 2 * dist_to_plane * self.c

        return self.ImageSource_Pos

    def _find_intersection_point(self, point1: Point, point2: Point) -> Point:
        """Find intersection point between IM-mic line segment and wall plane."""
        # Get first wall's plane coefficients (we only have one wall)
        wall = next(iter(self.walls.values()))
        
        # Get direction vector of the line
        l = point2.x - point1.x
        m = point2.y - point1.y
        n = point2.z - point1.z

        # Calculate intersection parameter k
        k = -(wall.plane_coeffs.a * point1.x + 
              wall.plane_coeffs.b * point1.y + 
              wall.plane_coeffs.c * point1.z +
              wall.plane_coeffs.d) / (wall.plane_coeffs.a * l + 
                                    wall.plane_coeffs.b * m + 
                                    wall.plane_coeffs.c * n)

        # Calculate intersection point
        interPos = Point(0.0, 0.0, 0.0)
        interPos.x = k * l + point1.x
        interPos.y = k * m + point1.y
        interPos.z = k * n + point1.z

        return interPos


class Wall:
    """
    Class defining a wall plane by 3 points
    """

    def __init__(self, posA: Point, posB: Point, posC: Point):
        self.plane_coeffs = Plane(posA, posB, posC)
        self.IMS = None
        self.node_positions = None  # simple attribute instead of property
        # Store wall boundaries
        self.corners = [posA, posB, posC]
        # Wall index for attenuation lookup
        self.wall_index = None

    def is_point_within_bounds(self, point: Point) -> bool:
        """Check if a point lies within the wall boundaries"""
        # Get vectors from first corner to other corners
        v1 = self.corners[1].subtract(self.corners[0])  # width vector
        v2 = self.corners[2].subtract(self.corners[0])  # height vector
        
        # Get vector from first corner to point
        vp = np.array([point.x - self.corners[0].x,
                       point.y - self.corners[0].y,
                       point.z - self.corners[0].z])
        
        # Project vp onto v1 and v2
        proj1 = np.dot(vp, v1) / np.dot(v1, v1)
        proj2 = np.dot(vp, v2) / np.dot(v2, v2)
        
        # Point is within bounds if both projections are between 0 and 1
        return (0 <= proj1 <= 1) and (0 <= proj2 <= 1)


class Point:
    """
    Class that defines a point in 3D cartesian coordinate system
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def getDistance(self, p):
        """
        Returns the Euclidean distance between two 3D positions in
        cartesian coordinates
        """
        return np.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2 + (self.z - p.z) ** 2)

    def subtract(self, p):
        """Returns the vector difference as numpy array"""
        return np.array([self.x - p.x, self.y - p.y, self.z - p.z], dtype=float)

    def equals(self, p):
        if (self.x == p.x) and (self.y == p.y) and (self.z == p.z):
            return True
        else:
            return False

    def to_array(self):
        """Explicit conversion to numpy array"""
        return np.array([self.x, self.y, self.z], dtype=float)

def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in radians."""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

def calculate_reflection_vector(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Calculate reflection vector using incident vector and surface normal."""
    # R = I - 2(I·N)N where I is incident vector, N is normal vector
    return incident - 2 * np.dot(incident, normal) * normal

def build_angle_mappings(room: Room) -> Dict[str, Dict[str, float]]:
    """Build dictionaries containing all source-to-node and node-to-node angles.
    
    Args:
        room: Room object containing geometry and source/mic positions
        
    Returns:
        Dict containing:
        - 'source_angles': Dict[str, float] - angles between source ray and wall normal
        - 'node_mappings': Dict[str, Dict[str, float]] - best reflection targets for each node
        - 'node_to_node_mappings': Dict[str, Dict[str, Dict[str, float]]] - node-to-node reflection mappings
    """
    mappings = {
        'source_angles': {},  # wall_id -> incident angle
        'node_mappings': {},   # wall_id -> {target_wall_id -> reflection angle}
        'node_to_node_mappings': {}  # wall_id -> {incident_wall_id -> {target_wall_id -> reflection angle}}
    }
    
    # Calculate source-to-node angles
    source_pos = np.array([room.source.srcPos.x, room.source.srcPos.y, room.source.srcPos.z])
    
    for wall_id, wall in room.walls.items():
        # Get wall normal and node position
        normal = np.array([wall.plane_coeffs.a, wall.plane_coeffs.b, wall.plane_coeffs.c])
        normal = normal / np.linalg.norm(normal)  # Normalize normal vector
        node_pos = np.array([wall.node_positions.x, wall.node_positions.y, wall.node_positions.z])
        
        # Calculate incident vector from source to node
        incident = node_pos - source_pos
        incident = incident / np.linalg.norm(incident)  # Normalize incident vector
        
        # Store incident angle
        angle = calculate_angle_between_vectors(incident, normal)
        mappings['source_angles'][wall_id] = angle
        
        # Calculate reflection vector
        reflection = calculate_reflection_vector(incident, normal)
        reflection = reflection / np.linalg.norm(reflection)  # Normalize reflection vector
        
        # Find best matching node for reflection
        node_scores = {}
        mappings['node_mappings'][wall_id] = {}
        
        for other_id, other_wall in room.walls.items():
            if other_id != wall_id:
                other_pos = np.array([other_wall.node_positions.x, 
                                    other_wall.node_positions.y,
                                    other_wall.node_positions.z])
                direction = other_pos - node_pos
                direction = direction / np.linalg.norm(direction)  # Normalize direction vector
                
                # Calculate angle between reflection vector and direction to other node
                reflection_angle = calculate_angle_between_vectors(reflection, direction)
                # If angle is obtuse, use the supplementary angle
                if reflection_angle > np.pi/2:
                    reflection_angle = np.pi - reflection_angle
                mappings['node_mappings'][wall_id][other_id] = reflection_angle
    
    # Calculate node-to-node reflection mappings
    for reflecting_wall_id, reflecting_wall in room.walls.items():
        mappings['node_to_node_mappings'][reflecting_wall_id] = {}
        reflecting_node_pos = np.array([reflecting_wall.node_positions.x, reflecting_wall.node_positions.y, reflecting_wall.node_positions.z])
        reflecting_normal = np.array([reflecting_wall.plane_coeffs.a, reflecting_wall.plane_coeffs.b, reflecting_wall.plane_coeffs.c])
        reflecting_normal = reflecting_normal / np.linalg.norm(reflecting_normal) # Normalize

        for incident_wall_id, incident_wall in room.walls.items():
            if incident_wall_id == reflecting_wall_id:
                continue
            
            mappings['node_to_node_mappings'][reflecting_wall_id][incident_wall_id] = {}
            incident_node_pos = np.array([incident_wall.node_positions.x, incident_wall.node_positions.y, incident_wall.node_positions.z])
            
            # Incident vector from incident_node to reflecting_node
            incident_vec_node = reflecting_node_pos - incident_node_pos
            if np.linalg.norm(incident_vec_node) == 0: continue # Avoid division by zero if nodes are at the same position
            incident_vec_node = incident_vec_node / np.linalg.norm(incident_vec_node)

            # Reflection vector at reflecting_node based on incidence from incident_node
            reflection_vec_node = calculate_reflection_vector(incident_vec_node, reflecting_normal)
            if np.linalg.norm(reflection_vec_node) == 0: continue
            reflection_vec_node = reflection_vec_node / np.linalg.norm(reflection_vec_node)

            for target_wall_id, target_wall in room.walls.items():
                if target_wall_id == reflecting_wall_id:
                    continue

                target_node_pos = np.array([target_wall.node_positions.x, target_wall.node_positions.y, target_wall.node_positions.z])
                direction_to_target = target_node_pos - reflecting_node_pos
                if np.linalg.norm(direction_to_target) == 0: continue
                direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)

                # Calculate angle between reflection vector and direction to target node
                node_reflection_angle = calculate_angle_between_vectors(reflection_vec_node, direction_to_target)
                
                # If angle is obtuse, use the supplementary angle (smallest angle)
                if node_reflection_angle > np.pi / 2:
                    node_reflection_angle = np.pi - node_reflection_angle
                
                mappings['node_to_node_mappings'][reflecting_wall_id][incident_wall_id][target_wall_id] = node_reflection_angle

    return mappings

def get_best_reflection_target(wall_id: str, mappings: Dict) -> str:
    """Get the wall ID that best matches the specular reflection direction.
    
    Args:
        wall_id: Current wall ID
        mappings: Angle mappings dictionary from build_angle_mappings()
        
    Returns:
        Wall ID of best reflection target
    """
    # if wall_id not in mappings['node_mappings']:
    #     return None
        
    # Find wall with smallest reflection angle
    angles = mappings['node_mappings'][wall_id]
    return min(angles.items(), key=lambda x: x[1])[0]

def get_best_reflection_targets(wall_id: str, mappings: Dict, num_targets: int = 2) -> List[str]:
    """Get the wall IDs that best match the specular reflection direction.
    
    Args:
        wall_id: Current wall ID
        mappings: Angle mappings dictionary from build_angle_mappings()
        num_targets: Number of best targets to return (default: 2)
        
    Returns:
        List of wall IDs of best reflection targets, sorted by angle (smallest first)
    """
    if wall_id not in mappings['node_mappings']:
        return []
        
    # Find walls with smallest reflection angles
    angles = mappings['node_mappings'][wall_id]
    # Sort by angle and get the specified number of targets
    sorted_targets = sorted(angles.items(), key=lambda x: x[1])[:num_targets]
    return [target[0] for target in sorted_targets]

def get_best_node2node_targets(wall_id: str, other_id : str, mappings: Dict, num_targets: int = 2) -> List[str]:
    """get the wall IDs that best math the node-node specular reflection direction, not source related"""
    if wall_id not in mappings['node_to_node_mappings']:
        return []

    # Find walls with the smallest reflection angles
    angles = mappings['node_to_node_mappings'][wall_id][other_id]
    # Sort by angle and get the specified number of targets
    sorted_targets = sorted(angles.items(), key=lambda x: x[1])[:num_targets]
    return [target[0] for target in sorted_targets]


def build_specular_matrices_from_angles(room: Room, increase_coef=0.0) -> Dict[str, np.ndarray]:
    """Build specular scattering matrices based on node-to-node angle mappings.
    
    For each wall and each incoming direction (column), calculates the best outgoing direction (row)
    based on specular reflection principle. The matrix values are derived from a modified
    diffuse scattering matrix, tuned by 'increase_coef'.

    If increase_coef is 0, the matrix is the standard diffuse scattering matrix.
    Otherwise:
    - The element at the best specular outgoing direction (row) for an incoming direction (column)
      is set to (2/size - 1) - increase_coef.
    - Other elements in that column are set to (2/size) + increase_coef / (size - 1).
    
    Args:
        room: Room object with walls and their normals
        increase_coef: Coefficient to adjust specular emphasis. Defaults to 0.0 (standard diffuse).
        
    Returns:
        Dict mapping wall IDs to their 5x5 scattering matrices
    """
    specular_mats = {}
    size = 5  # Matrix size (always 5x5 as we exclude current wall, N-1 for N=6)
    
    # Base values from diffuse matrix logic
    val_diag_equivalent = (2 / size - 1) - increase_coef
    val_offdiag_equivalent = (2 / size) + increase_coef / (size - 1)

    # Direction mappings for each wall (which other walls are its neighbors for scattering)
    direction_maps = {
        'north': ['south', 'west', 'east', 'ceiling', 'floor'],
        'south': ['north', 'west', 'east', 'ceiling', 'floor'],
        'west': ['south', 'north', 'east', 'ceiling', 'floor'],
        'east': ['south', 'north', 'west', 'ceiling', 'floor'],
        'ceiling': ['south', 'north', 'west', 'east', 'floor'],
        'floor': ['south', 'north', 'west', 'east', 'ceiling']
    }
    
    for wall_id, directions in direction_maps.items():
        # Get current wall's normal and node position (reflecting wall)
        reflecting_wall = room.walls[wall_id]
        reflecting_normal = np.array([reflecting_wall.plane_coeffs.a, reflecting_wall.plane_coeffs.b, reflecting_wall.plane_coeffs.c])
        reflecting_normal = reflecting_normal / np.linalg.norm(reflecting_normal)
        reflecting_node_pos = np.array([reflecting_wall.node_positions.x, reflecting_wall.node_positions.y, reflecting_wall.node_positions.z])
        
        mat = np.zeros((size, size)) # Initialize matrix
        
        # For each incoming direction (column index in_idx, corresponding to in_wall_id)
        for in_idx, in_wall_id in enumerate(directions):
            incoming_wall = room.walls[in_wall_id]
            incoming_node_pos = np.array([incoming_wall.node_positions.x, incoming_wall.node_positions.y, incoming_wall.node_positions.z])
            
            # Calculate incident vector (from incoming_node to reflecting_node)
            incident_vec = reflecting_node_pos - incoming_node_pos
            incident_vec = incident_vec / np.linalg.norm(incident_vec)
            
            # Calculate reflection vector using the law of reflection at the reflecting_wall
            reflection_vec = calculate_reflection_vector(incident_vec, reflecting_normal)
            
            # Find best matching outgoing direction (target_out_wall_id)
            best_target_out_idx = -1
            min_angle_diff_to_reflection = float('inf')
            
            # For each possible outgoing direction (row index out_idx, corresponding to target_out_wall_id)
            for out_idx, target_out_wall_id in enumerate(directions):
                target_out_wall = room.walls[target_out_wall_id]
                target_out_node_pos = np.array([target_out_wall.node_positions.x, target_out_wall.node_positions.y, target_out_wall.node_positions.z])
                
                # Calculate direction from reflecting_node to target_out_node
                direction_to_target_out = target_out_node_pos - reflecting_node_pos
                direction_to_target_out = direction_to_target_out / np.linalg.norm(direction_to_target_out)
                
                # Angle between the true reflection_vec and the direction_to_target_out
                angle_to_target = calculate_angle_between_vectors(reflection_vec, direction_to_target_out)
                
                if angle_to_target < min_angle_diff_to_reflection:
                    min_angle_diff_to_reflection = angle_to_target
                    best_target_out_idx = out_idx
            
            # Fill the column (in_idx) of the matrix
            for out_idx_fill in range(size):
                if out_idx_fill == best_target_out_idx:
                    mat[out_idx_fill, in_idx] = val_diag_equivalent
                else:
                    mat[out_idx_fill, in_idx] = val_offdiag_equivalent
        
        specular_mats[wall_id] = mat
    
    return specular_mats

def specular_matrices_test(room: Room, increase_coef=0.0):
    """Test and visualize the specular reflection matrices.
    
    Prints:
    1. The full matrix for each wall
    2. For each wall, lists the best reflection target for each incoming direction
    3. Verifies that each column has exactly one 'boosted' coefficient and sums to 1.0
    """
    matrices = build_specular_matrices_from_angles(room, increase_coef)
    size = 5 # N-1

    # Expected value for the "boosted" path (equivalent to modified diagonal)
    expected_boosted_val = (2 / size - 1) - increase_coef
    
    direction_maps = {
        'north': ['south', 'west', 'east', 'ceiling', 'floor'],
        'south': ['north', 'west', 'east', 'ceiling', 'floor'],
        'west': ['south', 'north', 'east', 'ceiling', 'floor'],
        'east': ['south', 'north', 'west', 'ceiling', 'floor'],
        'ceiling': ['south', 'north', 'west', 'east', 'floor'],
        'floor': ['south', 'north', 'west', 'east', 'ceiling']
    }
    
    print(f"\n=== Testing Specular Reflection Matrices (increase_coef: {increase_coef}) ===\n")
    
    for wall_id, matrix in matrices.items():
        print(f"\nWall: {wall_id.upper()}")
        print("Directions (columns=incoming, rows=outgoing):", direction_maps[wall_id])
        print("Matrix:")
        print(matrix)
        
        # Analyze reflection patterns
        print("\nReflection patterns (best outgoing row for each incoming column):")
        directions_for_wall = direction_maps[wall_id]
        
        # Pre-calculate best target row for each incoming column for this wall_id's matrix
        # This requires re-doing part of the logic from build_specular_matrices_from_angles
        # to find the expected best_target_out_idx for each in_idx
        reflecting_wall = room.walls[wall_id]
        reflecting_normal = np.array([reflecting_wall.plane_coeffs.a, reflecting_wall.plane_coeffs.b, reflecting_wall.plane_coeffs.c])
        reflecting_normal = reflecting_normal / np.linalg.norm(reflecting_normal)
        reflecting_node_pos = np.array([reflecting_wall.node_positions.x, reflecting_wall.node_positions.y, reflecting_wall.node_positions.z])

        boosted_coeffs_per_column = np.zeros(size, dtype=int)

        for in_idx, in_direction in enumerate(directions_for_wall):
            incoming_wall = room.walls[in_direction]
            incoming_node_pos = np.array([incoming_wall.node_positions.x, incoming_wall.node_positions.y, incoming_wall.node_positions.z])
            incident_vec = reflecting_node_pos - incoming_node_pos
            incident_vec = incident_vec / np.linalg.norm(incident_vec)
            reflection_vec = calculate_reflection_vector(incident_vec, reflecting_normal)
            
            expected_best_out_idx_for_test = -1
            min_angle_diff_for_test = float('inf')

            for out_idx_test, target_out_direction in enumerate(directions_for_wall):
                target_out_wall = room.walls[target_out_direction]
                target_out_node_pos = np.array([target_out_wall.node_positions.x, target_out_wall.node_positions.y, target_out_wall.node_positions.z])
                direction_to_target_out = target_out_node_pos - reflecting_node_pos
                direction_to_target_out = direction_to_target_out / np.linalg.norm(direction_to_target_out)
                angle_to_target = calculate_angle_between_vectors(reflection_vec, direction_to_target_out)
                if angle_to_target < min_angle_diff_for_test:
                    min_angle_diff_for_test = angle_to_target
                    expected_best_out_idx_for_test = out_idx_test
            
            # The row 'expected_best_out_idx_for_test' for column 'in_idx' should have the boosted value
            actual_val_at_boosted_path = matrix[expected_best_out_idx_for_test, in_idx]
            out_direction = directions_for_wall[expected_best_out_idx_for_test]
            print(f"  Incoming from {in_direction:7} -> Expected specular to {out_direction:7} (value: {actual_val_at_boosted_path:.4f})")
            
            if np.isclose(actual_val_at_boosted_path, expected_boosted_val):
                boosted_coeffs_per_column[in_idx] += 1
            
        # Verify matrix properties
        column_sums = np.sum(matrix, axis=0)
        
        print("\nVerification:")
        print(f"  Column sums (should all be 1.0): {column_sums}")
        for s in column_sums:
            assert np.isclose(s, 1.0), f"Column sum {s} is not 1.0"
            
        print(f"  Boosted coefficients per column (should all be 1, matching value {expected_boosted_val:.4f}): {boosted_coeffs_per_column}")
        for count in boosted_coeffs_per_column:
            assert count == 1, "Each column should have exactly one boosted coefficient"
        print("-" * 60)

def plot_reflection_comparison(room: Room, source_wall: str, ax=None):
    """Plot actual specular reflection and chosen best reflection paths.
    
    Args:
        room: Room object with walls and source/mic positions
        source_wall: Wall ID where reflection occurs (e.g., 'west')
        ax: Optional matplotlib 3D axis for plotting
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
    # Plot room first
    from plot_room import plot_room
    plot_room(room, ax)
    
    # Get source position and node position
    src_pos = room.source.srcPos
    node_pos = room.walls[source_wall].node_positions
    
    # Plot source to node line segment
    ax.plot([src_pos.x, node_pos.x], 
            [src_pos.y, node_pos.y], 
            [src_pos.z, node_pos.z], 
            'r--', linewidth=2, label='Source to Node')
    
    # Calculate actual specular reflection
    wall = room.walls[source_wall]
    normal = np.array([wall.plane_coeffs.a, wall.plane_coeffs.b, wall.plane_coeffs.c])
    normal = normal / np.linalg.norm(normal)
    
    # Calculate incident vector
    incident = np.array([node_pos.x - src_pos.x,
                        node_pos.y - src_pos.y,
                        node_pos.z - src_pos.z])
    incident = incident / np.linalg.norm(incident)
    
    # Calculate specular reflection vector
    reflection = calculate_reflection_vector(incident, normal)
    reflection = reflection / np.linalg.norm(reflection)
    
    # Calculate the maximum possible ray length based on room dimensions
    max_length = np.sqrt(room.x**2 + room.y**2 + room.z**2) * 2  # Diagonal of the room * 2
    
    # Plot specular reflection ray (extend it to max_length)
    end_point = np.array([node_pos.x, node_pos.y, node_pos.z]) + max_length * reflection
    ax.plot([node_pos.x, end_point[0]], 
            [node_pos.y, end_point[1]], 
            [node_pos.z, end_point[2]], 
            'g--', linewidth=2, label='Specular Reflection')
    
    # Get best reflection target from angle mappings
    best_target = get_best_reflection_target(source_wall, room.angle_mappings)
    target_pos = room.walls[best_target].node_positions
    print("Best reflection target for wall", source_wall, "is", best_target, "at position:", target_pos)
    # Plot actual chosen reflection path
    ax.plot([node_pos.x, target_pos.x], 
            [node_pos.y, target_pos.y], 
            [node_pos.z, target_pos.z], 
            'b--', linewidth=2, label='Chosen Reflection')
    
    # Add legend and labels
    ax.legend()
    ax.set_title(f'Reflection Paths from {source_wall.upper()} wall')
    
    # Set view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    return ax


class Plane:
    """
    Class and helper functions defining a 3D plane
    """

    def __init__(self, posA, posB, posC):
        # plane represented by ax + by + cz + d = 0 and its normal vector
        # posA, posB and posC are 3 points on a plane

        # find vector normal to the plane
        arr1 = posB.subtract(posA)
        arr2 = posC.subtract(posA)
        self.normal = np.cross(arr1, arr2)

        assert np.dot(self.normal, arr1) == 0.0, "normal vector not right"

        # scalar component
        self.d = np.dot(-self.normal, [posA.x, posA.y, posA.z])
        self.a = self.normal[0]
        self.b = self.normal[1]
        self.c = self.normal[2]

def t_est_angle_calculations(test_room):
    """Test function to verify the correctness of angle calculations."""
    
    # Calculate angle mappings
    mappings = build_angle_mappings(test_room)
    
    # Test 1: Verify incident angle equals reflection angle for each wall
    print("\nTest 1: Incident vs Reflection Angles")
    for wall_id, wall in test_room.walls.items():
        # Get wall normal
        normal = np.array([wall.plane_coeffs.a, wall.plane_coeffs.b, wall.plane_coeffs.c])
        normal = normal / np.linalg.norm(normal)
        
        # Get node position and source position
        node_pos = np.array([wall.node_positions.x, wall.node_positions.y, wall.node_positions.z])
        source_pos = np.array([test_room.source.srcPos.x, test_room.source.srcPos.y, test_room.source.srcPos.z])
        
        # Calculate incident vector and its reflection
        incident = node_pos - source_pos
        incident = incident / np.linalg.norm(incident)
        reflection = calculate_reflection_vector(incident, normal)
        
        # Calculate angles from normal (both should be acute angles)
        incident_angle = calculate_angle_between_vectors(incident, normal)
        # If incident_angle > π/2, use the supplementary angle
        if incident_angle > np.pi/2:
            incident_angle = np.pi - incident_angle
        
        # For reflection angle, we need to ensure we're measuring it the same way as incident
        reflection_angle = calculate_angle_between_vectors(reflection, normal)
        if reflection_angle > np.pi/2:
            reflection_angle = np.pi - reflection_angle
        
        print(f"{wall_id}:")
        print(f"  Incident angle from normal: {np.degrees(incident_angle):.2f}°")
        print(f"  Reflection angle from normal: {np.degrees(reflection_angle):.2f}°")
        print(f"  Difference: {np.degrees(abs(incident_angle - reflection_angle)):.6f}°")
        assert np.abs(incident_angle - reflection_angle) < 1e-10, "Incident angle should equal reflection angle"
    
    # Test 2: Verify symmetry of reflection paths
    print("\nTest 2: Path Symmetry")
    test_pairs = [
        ('north', 'south'),
        ('east', 'west'),
        ('ceiling', 'floor')
    ]
    
    for wall1, wall2 in test_pairs:
        # Get angles for path: source -> wall1 -> wall2
        path1_angle1 = mappings['source_angles'][wall1]
        path1_angle2 = mappings['node_mappings'][wall1][wall2]
        
        # Get angles for path: source -> wall2 -> wall1
        path2_angle1 = mappings['source_angles'][wall2]
        path2_angle2 = mappings['node_mappings'][wall2][wall1]
        
        print(f"\n{wall1}-{wall2} pair:")
        print(f"  Path 1 (source->{wall1}->{wall2}): {np.degrees(path1_angle1 + path1_angle2):.2f}°")
        print(f"  Path 2 (source->{wall2}->{wall1}): {np.degrees(path2_angle1 + path2_angle2):.2f}°")
        print(f"  Difference: {np.degrees(abs((path1_angle1 + path1_angle2) - (path2_angle1 + path2_angle2))):.6f}°")
    
    # Test 3: Verify best reflection targets are physically correct
    print("\nTest 3: Physical Correctness of Best Reflection Targets")
    expected_pairs = {
        'north': 'south',
        'south': 'west',
        'east': 'west',
        'west': 'east',
        'ceiling': 'floor',
        'floor': 'west'
    }
    
    for wall_id, expected_target in expected_pairs.items():
        best_target = get_best_reflection_target(wall_id, mappings)
        print(f"{wall_id}: Expected {expected_target}, Got {best_target}")
        assert best_target == expected_target, f"Unexpected reflection target for {wall_id}"

def get_image_sources(room: Room, max_order: int) -> List[Dict]:
    """
    Calculates all image sources up to a specified order for a cuboid room,
    ensuring that the reflection paths are geometrically valid.

    Args:
        room (Room): The room object containing walls, source, and mic positions.
        max_order (int): The maximum reflection order to calculate.

    Returns:
        List[Dict]: A list of valid image source information, where each entry is a
                    dictionary with:
                    - 'position': Point - Image source position
                    - 'order': int - Reflection order
                    - 'path': List[str] - Path sequence (e.g., ['s', 'west', 'floor'])
                    - 'last_bounce': Point - Last intersection point (on the last wall)
    """
    valid_image_sources = []
    
    # Initial source
    q = [{'position': room.source.srcPos, 'order': 0, 'path': ['s']}]
    
    visited_paths = {('s',)}

    for order in range(1, max_order + 1):
        next_q = []
        for source in q:
            current_pos = source['position']
            current_path = source['path']
            
            for wall_label, wall in room.walls.items():
                if len(current_path) > 1 and wall_label == current_path[-1]:
                    continue

                new_path_tuple = tuple(current_path + [wall_label])
                if new_path_tuple in visited_paths:
                    continue
                
                visited_paths.add(new_path_tuple)

                # Tentatively calculate the next image source position
                temp_image_source = current_pos
                path_for_calc = new_path_tuple[1:] # Path without 's'
                
                # Reflect through all walls in the new path to get the final image source
                final_image_source_pos = room.source.srcPos
                for p_wall_id in path_for_calc:
                    p_wall = room.walls[p_wall_id]
                    a, b, c, d = p_wall.plane_coeffs.a, p_wall.plane_coeffs.b, p_wall.plane_coeffs.c, p_wall.plane_coeffs.d
                    norm_sq = a**2 + b**2 + c**2
                    if norm_sq == 0: continue
                    dist_to_plane = (a * final_image_source_pos.x + b * final_image_source_pos.y + c * final_image_source_pos.z + d) / norm_sq
                    final_image_source_pos = Point(
                        final_image_source_pos.x - 2 * dist_to_plane * a,
                        final_image_source_pos.y - 2 * dist_to_plane * b,
                        final_image_source_pos.z - 2 * dist_to_plane * c
                    )

                # Validate the path by checking each bounce point
                is_path_valid = True
                last_bounce_point = None  # NEW: Store the last bounce point
                
                # The principle: trace from the receiver back to the source,
                # ensuring each bounce is valid.
                
                # The logic here is adapted directly from the user's proven ISMCalculator
                
                current_point = room.micPos # Start the backward trace from the actual microphone
                current_image = final_image_source_pos
                
                walls_reversed = list(reversed(path_for_calc))

                for i, wall_id in enumerate(walls_reversed):
                    wall = room.walls[wall_id]
                    
                    # Create a dummy ImageSource instance just for the intersection calculation
                    img_source_calc = ImageSource({'wall': wall}, current_image, current_point)
                    intersection = img_source_calc._find_intersection_point(current_point, current_image)
                    
                    if not wall.is_point_within_bounds(intersection):
                        is_path_valid = False
                        break
                    
                    # NEW: Store the first intersection (which is the last bounce in forward time)
                    if i == 0:
                        last_bounce_point = intersection

                    # Prepare for the next iteration in the backward trace
                    current_point = intersection
                    if i < len(walls_reversed) - 1:
                        # We need the image source for the path *before* this bounce
                        prev_path_for_calc = path_for_calc[:-(i+1)]
                        
                        # Recalculate the image source for the shorter path
                        img_src_for_next_bounce = room.source.srcPos
                        for p_wall_id in prev_path_for_calc:
                            p_wall = room.walls[p_wall_id]
                            a, b, c, d = p_wall.plane_coeffs.a, p_wall.plane_coeffs.b, p_wall.plane_coeffs.c, p_wall.plane_coeffs.d
                            norm_sq = a**2 + b**2 + c**2
                            if norm_sq == 0: continue
                            dist_to_plane = (a * img_src_for_next_bounce.x + b * img_src_for_next_bounce.y + c * img_src_for_next_bounce.z + d) / norm_sq
                            img_src_for_next_bounce = Point(
                                img_src_for_next_bounce.x - 2 * dist_to_plane * a,
                                img_src_for_next_bounce.y - 2 * dist_to_plane * b,
                                img_src_for_next_bounce.z - 2 * dist_to_plane * c
                            )
                        current_image = img_src_for_next_bounce


                if is_path_valid:
                    new_source = {
                        'position': final_image_source_pos,
                        'order': order,
                        'path': list(new_path_tuple),
                        'last_bounce': last_bounce_point  # NEW: Add last bounce point
                    }
                    valid_image_sources.append(new_source)
                    next_q.append(new_source)

        q = next_q

    return valid_image_sources

if __name__ == "__main__":
    # Run the angle calculation tests


    # Create visualization of reflection paths
    test_room = Room(9, 7, 4)
    test_room.set_microphone(0.5, 0.5, 1.5)
    test_room.set_source(8.5, 6, 2, signal="")

    # t_est_angle_calculations(test_room)

    # Create single plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_reflection_comparison(test_room, 'floor', ax)
    plot_reflection_comparison(test_room, 'ceiling', ax)
    plot_reflection_comparison(test_room, 'south', ax)
    plot_reflection_comparison(test_room, 'north', ax)
    plot_reflection_comparison(test_room, 'east', ax)
    plot_reflection_comparison(test_room, 'west', ax)

    plt.tight_layout()
    plt.show()

    # specular_matrices_test(test_room, increase_coef=0.2) # Test with a non-zero increase_coef
    # specular_matrices_test(test_room, increase_coef=0.0) # Test with zero increase_coef (should be diffuse)