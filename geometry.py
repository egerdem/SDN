from __future__ import annotations  # This allows forward references in type hints
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    def set_microphone(self, mx, my, mz):
        self.mx = mx
        self.my = my
        self.mz = mz
        self.micPos = Point(mx, my, mz)

    def set_source(self, sx, sy, sz, signal, Fs=44100):
        self.source = Source(sx, sy, sz, signal, Fs)
        self.srcPos = Point(sx, sy, sz)
        # Calculate SDN node positions (first-order reflection points)
        self._calculate_sdn_nodes()
        # Calculate angle mappings after nodes are set up
        self.angle_mappings = build_angle_mappings(self)

    def _calculate_sdn_nodes(self):
        """Calculate fixed SDN node positions (first-order reflection points)"""
        for wall_label, wall in self.walls.items():
            # Calculate image source for "wall"
            img_source = ImageSource({wall_label: wall}, self.srcPos, self.micPos)
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
        normal = normal / np.linalg.norm(normal)
        node_pos = np.array([wall.node_positions.x, wall.node_positions.y, wall.node_positions.z])
        
        # Calculate incident vector from source to node
        incident = node_pos - source_pos
        incident = incident / np.linalg.norm(incident)
        
        # Store incident angle
        angle = calculate_angle_between_vectors(incident, normal)
        mappings['source_angles'][wall_id] = angle
        
        # Calculate reflection vector
        reflection = calculate_reflection_vector(incident, normal)
        
        # Find best matching node for reflection
        node_scores = {}
        mappings['node_mappings'][wall_id] = {}
        
        for other_id, other_wall in room.walls.items():
            if other_id != wall_id:
                other_pos = np.array([other_wall.node_positions.x, 
                                    other_wall.node_positions.y,
                                    other_wall.node_positions.z])
                direction = other_pos - node_pos
                direction = direction / np.linalg.norm(direction)
                
                # Calculate angle between reflection vector and direction to other node
                reflection_angle = calculate_angle_between_vectors(reflection, direction)
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
    return max(angles.items(), key=lambda x: x[1])[0]

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

def test_specular_matrices(room: Room):
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

if __name__ == "__main__":
    # Create a test room
    test_room = Room(6, 4, 3)  # 6m x 4m x 3m room
    test_room.set_microphone(4, 3, 1.5)  # Mic position
    test_room.set_source(2, 2, 1.7, signal="")  # Source position (signal not needed for this test)
    
    # Run the test
    test_specular_matrices(test_room)

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
