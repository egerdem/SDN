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
            import frequency as ff
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
    """
    mappings = {
        'source_angles': {},  # wall_id -> incident angle
        'node_mappings': {}   # wall_id -> {target_wall_id -> reflection angle}
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
    
    return mappings

def get_best_reflection_target(wall_id: str, mappings: Dict) -> str:
    """Get the wall ID that best matches the specular reflection direction.
    
    Args:
        wall_id: Current wall ID
        mappings: Angle mappings dictionary from build_angle_mappings()
        
    Returns:
        Wall ID of best reflection target
    """
    if wall_id not in mappings['node_mappings']:
        return None
        
    # Find wall with smallest reflection angle
    angles = mappings['node_mappings'][wall_id]
    return min(angles.items(), key=lambda x: x[1])[0]

def build_specular_matrices_from_angles(room: Room) -> Dict[str, np.ndarray]:
    """Build specular scattering matrices based on node-to-node angle mappings.
    
    For each wall and each incoming direction, calculates the best outgoing direction
    based on specular reflection principle. Creates a 5x5 matrix where:
    - For each incoming direction (column), the best outgoing direction (row) gets specular coefficient (0.8)
    - Other outgoing directions get diffuse coefficient ((1-0.8)/4 = 0.05)
    
    Args:
        room: Room object with walls and their normals
        
    Returns:
        Dict mapping wall IDs to their 5x5 specular matrices
    """
    specular_mats = {}
    specular_coeff = 0.8
    size = 5  # Matrix size (always 5x5 as we exclude current wall)
    diffuse_coeff = (1.0 - specular_coeff) / (size - 1)
    
    # Direction mappings for each wall (which walls to exclude)
    direction_maps = {
        'north': ['south', 'west', 'east', 'ceiling', 'floor'],
        'south': ['north', 'west', 'east', 'ceiling', 'floor'],
        'west': ['south', 'north', 'east', 'ceiling', 'floor'],
        'east': ['south', 'north', 'west', 'ceiling', 'floor'],
        'ceiling': ['south', 'north', 'west', 'east', 'floor'],
        'floor': ['south', 'north', 'west', 'east', 'ceiling']
    }
    
    for wall_id, directions in direction_maps.items():
        # Get current wall's normal and node position
        wall = room.walls[wall_id]
        normal = np.array([wall.plane_coeffs.a, wall.plane_coeffs.b, wall.plane_coeffs.c])
        normal = normal / np.linalg.norm(normal)
        node_pos = np.array([wall.node_positions.x, wall.node_positions.y, wall.node_positions.z])
        
        # Create base matrix with diffuse coefficients
        mat = np.ones((size, size)) * diffuse_coeff
        
        # For each incoming direction (column)
        for in_idx, in_wall_id in enumerate(directions):
            in_wall = room.walls[in_wall_id]
            in_pos = np.array([in_wall.node_positions.x, in_wall.node_positions.y, in_wall.node_positions.z])
            
            # Calculate incident vector (from incoming node to current node)
            incident = node_pos - in_pos
            incident = incident / np.linalg.norm(incident)
            
            # Calculate incident angle with wall normal
            incident_angle = calculate_angle_between_vectors(incident, normal)
            
            # Calculate reflection vector using the law of reflection
            reflection = calculate_reflection_vector(incident, normal)
            
            # Find best matching outgoing direction
            best_out_idx = 0
            min_angle_diff = float('inf')
            
            # For each possible outgoing direction
            for out_idx, out_wall_id in enumerate(directions):
                out_wall = room.walls[out_wall_id]
                out_pos = np.array([out_wall.node_positions.x, out_wall.node_positions.y, out_wall.node_positions.z])
                
                # Calculate direction to outgoing node
                direction = out_pos - node_pos
                direction = direction / np.linalg.norm(direction)
                
                # Calculate angle between reflection vector and direction to other node
                out_angle = calculate_angle_between_vectors(reflection, direction)
                
                # For perfect specular reflection, we want the outgoing angle to match the incident angle
                angle_diff = abs(out_angle)
                
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_out_idx = out_idx
            
            # Set specular coefficient for best outgoing direction
            mat[best_out_idx, in_idx] = specular_coeff
        
        specular_mats[wall_id] = mat
    
    return specular_mats

def specular_matrices_test(room: Room):
    """Test and visualize the specular reflection matrices.
    
    Prints:
    1. The full matrix for each wall
    2. For each wall, lists the best reflection target for each incoming direction
    3. Verifies that each column has exactly one specular coefficient
    """
    matrices = build_specular_matrices_from_angles(room)
    
    # Direction mappings (same as in build_specular_matrices_from_angles)
    direction_maps = {
        'north': ['south', 'west', 'east', 'ceiling', 'floor'],
        'south': ['north', 'west', 'east', 'ceiling', 'floor'],
        'west': ['south', 'north', 'east', 'ceiling', 'floor'],
        'east': ['south', 'north', 'west', 'ceiling', 'floor'],
        'ceiling': ['south', 'north', 'west', 'east', 'floor'],
        'floor': ['south', 'north', 'west', 'east', 'ceiling']
    }
    
    specular_coeff = 0.8  # Should match the value in build_specular_matrices_from_angles
    
    print("\n=== Testing Specular Reflection Matrices ===\n")
    
    for wall_id, matrix in matrices.items():
        print(f"\nWall: {wall_id.upper()}")
        print("Directions (columns=incoming, rows=outgoing):", direction_maps[wall_id])
        print("Matrix:")
        print(matrix)
        
        # Analyze reflection patterns
        print("\nReflection patterns:")
        directions = direction_maps[wall_id]
        for in_idx, in_direction in enumerate(directions):
            # Find the specular reflection direction (row with coefficient 0.8)
            out_idx = np.where(np.abs(matrix[:, in_idx] - specular_coeff) < 1e-6)[0][0]
            out_direction = directions[out_idx]
            print(f"  Incoming from {in_direction:7} -> Specular reflection to {out_direction:7}")
            
        # Verify matrix properties
        column_sums = np.sum(matrix, axis=0)
        specular_counts = np.sum(np.abs(matrix - specular_coeff) < 1e-6, axis=0)
        
        print("\nVerification:")
        print(f"  Column sums (should all be 1.0): {column_sums}")
        print(f"  Specular coefficients per column (should all be 1): {specular_counts}")
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

def test_angle_calculations():
    """Test function to verify the correctness of angle calculations."""
    # Create a simple test room
    test_room = Room(6, 4, 3)  # 6m x 4m x 3m room
    test_room.set_microphone(4, 3, 1.5)  # Mic position
    test_room.set_source(2, 2, 1.7, signal="")  # Source position
    
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
        'south': 'north',
        'east': 'west',
        'west': 'east',
        'ceiling': 'floor',
        'floor': 'ceiling'
    }
    
    for wall_id, expected_target in expected_pairs.items():
        best_target = get_best_reflection_target(wall_id, mappings)
        print(f"{wall_id}: Expected {expected_target}, Got {best_target}")
        assert best_target == expected_target, f"Unexpected reflection target for {wall_id}"

if __name__ == "__main__":
    # Run the angle calculation tests
    test_angle_calculations()
    
    # Create visualization of reflection paths
    test_room = Room(6, 4, 3)
    test_room.set_microphone(4, 3, 1.5)
    test_room.set_source(2, 2, 1.7, signal="")
    
    # Create single plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_reflection_comparison(test_room, 'south', ax)
    plt.tight_layout()
    plt.show()