import numpy as np
from typing import Dict, Tuple
from geometry import Point, Room, Wall
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')

def directions_for_north():
    # Node is 'north' -> exclude 'north'
    return ['south','west','east','ceiling','floor']

def reflection_map_north():
    """
    Indices: 0='south', 1='west', 2='east', 3='ceiling', 4='floor'
    Specular flipping:
      south->south
      west->east
      east->west
      ceiling->floor
      floor->ceiling
    """
    return {
        0:0,  # south -> south
        1:2,  # west  -> east
        2:1,  # east  -> west
        3:4,  # ceiling -> floor
        4:3,  # floor   -> ceiling
    }

def directions_for_south():
    # Node is 'south' -> exclude 'south'
    return ['north','west','east','ceiling','floor']

def reflection_map_south():
    """
    Indices: 0='north', 1='west', 2='east', 3='ceiling', 4='floor'
    Flip left-right, up-down, keep front-back:
      north->north
      west->east
      east->west
      ceiling->floor
      floor->ceiling
    """
    return {
        0:0,
        1:2,
        2:1,
        3:4,
        4:3,
    }

def directions_for_west():
    # Node is 'west' -> exclude 'west'
    return ['south','north','east','ceiling','floor']

def reflection_map_west():
    """
    Indices: 0='south', 1='north', 2='east', 3='ceiling', 4='floor'
    'west' node => wave from 'east' remains 'east'?
    we do front-back, up-down flips, so:
      south->north, north->south, ceiling->floor, floor->ceiling, east->east
    """
    return {
        0:1,  # south->north
        1:0,  # north->south
        2:2,  # east->east
        3:4,  # ceiling->floor
        4:3,  # floor->ceiling
    }

def directions_for_east():
    # Node is 'east' -> exclude 'east'
    return ['south','north','west','ceiling','floor']

def reflection_map_east():
    """
    Indices: 0='south', 1='north', 2='west', 3='ceiling', 4='floor'
    'east' node => wave from 'west' remains 'west'?
    again front-back, up-down flips:
      south->north, north->south, west->west, ceiling->floor, floor->ceiling
    """
    return {
        0:1,  # south->north
        1:0,  # north->south
        2:2,  # west->west
        3:4,  # ceiling->floor
        4:3,  # floor->ceiling
    }

def directions_for_ceiling():
    # Node is 'ceiling' -> exclude 'ceiling'
    return ['south','north','west','east','floor']

def reflection_map_ceiling():
    """
    Indices: 0='south', 1='north', 2='west', 3='east', 4='floor'
    For 'ceiling' node, we do front-back, left-right flips, keep floor->floor?
    Actually we want floor->floor to remain the same.
    So:
      south->north, north->south, west->east, east->west, floor->floor
    """
    return {
        0:1,  # south->north
        1:0,  # north->south
        2:3,  # west->east
        3:2,  # east->west
        4:4,  # floor->floor
    }

def directions_for_floor():
    # Node is 'floor' -> exclude 'floor'
    return ['south','north','west','east','ceiling']

def reflection_map_floor():
    """
    Indices: 0='south', 1='north', 2='west', 3='east', 4='ceiling'
    For 'floor' node, we want the wave from south->north, north->south, west->east, east->west, ceiling->ceiling
    """
    return {
        0:1,  # south->north
        1:0,  # north->south
        2:3,  # west->east
        3:2,  # east->west
        4:4,  # ceiling->ceiling
    }

def create_specular_scattering_matrix(size, reflection_map):
    """
    Returns a permutation (specular) scattering matrix of shape (size, size).
    reflection_map: dict from incoming_index -> outgoing_index
    """
    # mat = np.zeros((size, size))
    specular_coeff = 0.8
    mat = np.ones((size, size)) * (1.0 - specular_coeff) / (size - 1)
    for i in range(size):
        j = reflection_map[i]
        mat[j, i] = specular_coeff
    return mat

def build_specular_matrices():
    """
    Build a dictionary containing a (5x5) specular scattering matrix for each wall:
    'south','north','west','east','ceiling','floor'.
    """
    specular_mats = {}

    # north
    dn = directions_for_north()
    map_n = reflection_map_north()
    mat_n = create_specular_scattering_matrix(len(dn), map_n)
    specular_mats['north'] = mat_n

    # south
    ds = directions_for_south()
    map_s = reflection_map_south()
    mat_s = create_specular_scattering_matrix(len(ds), map_s)
    specular_mats['south'] = mat_s

    # west
    dw = directions_for_west()
    map_w = reflection_map_west()
    mat_w = create_specular_scattering_matrix(len(dw), map_w)
    specular_mats['west'] = mat_w

    # east
    de = directions_for_east()
    map_e = reflection_map_east()
    mat_e = create_specular_scattering_matrix(len(de), map_e)
    specular_mats['east'] = mat_e

    # ceiling
    dc = directions_for_ceiling()
    map_c = reflection_map_ceiling()
    mat_c = create_specular_scattering_matrix(len(dc), map_c)
    specular_mats['ceiling'] = mat_c

    # floor
    df = directions_for_floor()
    map_f = reflection_map_floor()
    mat_f = create_specular_scattering_matrix(len(df), map_f)
    specular_mats['floor'] = mat_f

    return specular_mats

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
    return min(angles.items(), key=lambda x: x[1])[0]

if __name__ == '__main__':
    from analysis import plot_room as pp
    
    # First print the specular matrices as before
    spec_mats = build_specular_matrices()
    for node_label, mat in spec_mats.items():
        print(f"\nSpecular matrix for '{node_label}':\n{mat}")

    """# Create a test room
    print("\n=== Testing Angle Mappings with Sample Room ===")
    test_room = Room(6, 4, 3)  # 6m x 4m x 3m room
    test_room.set_microphone(4, 3, 1.5)  # Mic position
    test_room.set_source(2, 2, 1.7, signal="")  # Source position
    
    # Calculate angle mappings
    mappings = build_angle_mappings(test_room)
    
    # Print source-to-node angles
    print("\nSource to Node Angles (radians):")
    for wall_id, angle in mappings['source_angles'].items():
        print(f"{wall_id}: {angle:.3f}")
    
    # Print best reflection targets for each wall
    print("\nBest Reflection Targets:")
    for wall_id in test_room.walls:
        best_target = get_best_reflection_target(wall_id, mappings)
        print(f"From {wall_id} -> {best_target} (angle: {mappings['node_mappings'][wall_id][best_target]:.3f} rad)")
    
    # Create visualization using plot_room
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Basic room with nodes
    ax1 = fig.add_subplot(121, projection='3d')
    pp.plot_room(test_room, ax=ax1)
    ax1.set_title('Room with Nodes')
    
    # Plot 2: Room with reflection paths
    ax2 = fig.add_subplot(122, projection='3d')
    pp.plot_room(test_room, ax=ax2)
    
    # Add reflection paths
    colors = plt.cm.rainbow(np.linspace(0, 1, len(test_room.walls)))
    for wall_id, wall, color in zip(test_room.walls.keys(), test_room.walls.values(), colors):
        node_pos = wall.node_positions
        
        # Plot source to node ray
        ax2.plot([test_room.source.srcPos.x, node_pos.x],
                [test_room.source.srcPos.y, node_pos.y],
                [test_room.source.srcPos.z, node_pos.z],
                '--', color=color, alpha=0.7, label=f'Source to {wall_id}')
        
        # Get best reflection target and plot
        best_target = get_best_reflection_target(wall_id, mappings)
        target_pos = test_room.walls[best_target].node_positions
        
        ax2.plot([node_pos.x, target_pos.x],
                [node_pos.y, target_pos.y],
                [node_pos.z, target_pos.z],
                ':', color=color, alpha=0.5)
        
        # Add text annotations for nodes
        ax2.text(node_pos.x, node_pos.y, node_pos.z, 
                wall_id, fontsize=8)
    
    ax2.set_title('Room with Reflection Paths')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Analyze path asymmetry
    print("\n=== Analyzing Path Asymmetry ===")
    test_cases = [
        ('north', 'south'),
        ('floor', 'ceiling'),
        ('east', 'west')
    ]
    
    for wall1, wall2 in test_cases:
        print(f"\nAnalyzing {wall1}-{wall2} paths:")
        # Source -> Wall1 -> Wall2
        src_wall1_angle = mappings['source_angles'][wall1]
        wall1_wall2_angle = mappings['node_mappings'][wall1][wall2]
        
        # Source -> Wall2 -> Wall1
        src_wall2_angle = mappings['source_angles'][wall2]
        wall2_wall1_angle = mappings['node_mappings'][wall2][wall1]
        
        print(f"Path 1: Source -> {wall1} -> {wall2}")
        print(f"  Source->{wall1} angle: {src_wall1_angle:.3f} rad")
        print(f"  {wall1}->{wall2} angle: {wall1_wall2_angle:.3f} rad")
        
        print(f"Path 2: Source -> {wall2} -> {wall1}")
        print(f"  Source->{wall2} angle: {src_wall2_angle:.3f} rad")
        print(f"  {wall2}->{wall1} angle: {wall2_wall1_angle:.3f} rad")
        
        # Calculate total path angles
        total_angle1 = src_wall1_angle + wall1_wall2_angle
        total_angle2 = src_wall2_angle + wall2_wall1_angle
        print(f"Total angles: Path1={total_angle1:.3f} rad, Path2={total_angle2:.3f} rad")
        print(f"Angle difference: {abs(total_angle1 - total_angle2):.3f} rad")
    
    plt.tight_layout()
    plt.show()
"""


# ============================================================================
# Experimental SDN Functions (moved from sdn_core.py)
# ============================================================================

def random_wall_mapping(walls):
    """
    ONLY USED if specular_source_injection_random flag is True.
    Experiment for: RANDOM source injection, not used by the main SW-SDN
    
    Given a list of unique wall‐IDs, return a dict that maps each wall
    to a different one, with no wall mapped to itself.
    
    >>> walls = ["e", "w", "s", "f", "n", "c"]
    >>> random_wall_mapping(walls)
    {'e': 'n', 'w': 'c', 's': 'e', 'f': 'w', 'n': 's', 'c': 'f'}
    """
    import random
    
    if len(set(walls)) != len(walls):
        raise ValueError("Wall IDs must be unique")
    
    while True:
        shuffled = walls[:]        # copy
        random.shuffle(shuffled)   # in-place random permutation
        
        # keep the permutation only if nothing stayed in place
        if all(orig != new for orig, new in zip(walls, shuffled)):
            return dict(zip(walls, shuffled))


def scattering_matrix_crate(increase_coef=0.2):
    """
    ONLY USED if scattering_matrix_update_coef flag is not None.
    Experiment for: increasing the off-diagonal elements of the scattering matrix
    and decreasing the others accordingly to compensate for the overall energy.
    
    Create a modified scattering matrix with adjusted diagonal and off-diagonal elements.
    
    Args:
        increase_coef (float): Coefficient to adjust the matrix elements
        
    Returns:
        np.ndarray: Modified scattering matrix
    """
    c = increase_coef
    original_matrix = (2 / 5) * np.ones((5, 5)) - np.eye(5)
    adjusted_matrix = np.copy(original_matrix)
    
    # Decrease diagonal elements by c
    np.fill_diagonal(adjusted_matrix, adjusted_matrix.diagonal() - c)
    
    # Increase only off-diagonal elements by c/4
    off_diagonal_mask = np.ones(adjusted_matrix.shape, dtype=bool)
    np.fill_diagonal(off_diagonal_mask, False)
    
    # Add c/4 to only the off-diagonal elements
    adjusted_matrix[off_diagonal_mask] += (c / 4)
    return adjusted_matrix