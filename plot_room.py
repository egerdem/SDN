import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import pyroomacoustics as pra

def plot_room(room, ax=None):
    """Plot room geometry with source, mic, and walls."""
    if ax is None:
        plt.close('all')  # Close any existing plots
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot source and microphone
    ax.scatter(room.source.sx, room.source.sy, room.source.sz, 
              color='green', s=100, label='Source')
    ax.scatter(room.mx, room.my, room.mz, 
              color='red', s=100, label='Microphone')
    
    # Define colors for each wall
    wall_colors = {
        'floor': 'lightgray',
        'ceiling': 'darkgray',
        'west': 'lightblue',
        'east': 'lightgreen',
        'north': 'pink',
        'south': 'wheat'
    }
    
    # Plot walls using plane equations
    for wall_label, wall in room.walls.items():
        # Find the largest coefficient to determine which variable to solve for
        coeffs = [abs(wall.plane_coeffs.a), 
                 abs(wall.plane_coeffs.b), 
                 abs(wall.plane_coeffs.c)]
        max_coeff_idx = np.argmax(coeffs)
        
        # Create a grid of points based on the dominant coefficient
        if max_coeff_idx == 2:  # z is dominant
            xx, yy = np.meshgrid(np.linspace(0, room.x, 10),
                               np.linspace(0, room.y, 10))
            zz = (-wall.plane_coeffs.a * xx - wall.plane_coeffs.b * yy 
                  - wall.plane_coeffs.d) / wall.plane_coeffs.c
        
        elif max_coeff_idx == 0:  # x is dominant
            yy, zz = np.meshgrid(np.linspace(0, room.y, 10),
                               np.linspace(0, room.z, 10))
            xx = (-wall.plane_coeffs.b * yy - wall.plane_coeffs.c * zz 
                  - wall.plane_coeffs.d) / wall.plane_coeffs.a
            
        else:  # y is dominant
            xx, zz = np.meshgrid(np.linspace(0, room.x, 10),
                               np.linspace(0, room.z, 10))
            yy = (-wall.plane_coeffs.a * xx - wall.plane_coeffs.c * zz 
                  - wall.plane_coeffs.d) / wall.plane_coeffs.b
            
        surf = ax.plot_surface(xx, yy, zz, alpha=0.1, 
                             color=wall_colors[wall_label])
        
        # Add wall label at the center of each wall, but offset from the surface
        center_x = xx.mean()
        center_y = yy.mean()
        center_z = zz.mean()
        
        # Calculate offset direction based on wall normal vector
        normal = np.array([wall.plane_coeffs.a, wall.plane_coeffs.b, wall.plane_coeffs.c])
        normal = normal / np.linalg.norm(normal)  # Normalize the vector
        offset = 0  # Adjust this value to control label distance from wall
        
        # Apply offset to center position
        label_x = center_x + normal[0] * offset
        label_y = center_y + normal[1] * offset
        label_z = center_z + normal[2] * offset
        
        ax.text(label_x, label_y, label_z, wall_label, 
                horizontalalignment='center', verticalalignment='center')
        
        # Plot node positions
        node_pos = wall.node_positions
        ax.scatter(node_pos.x, node_pos.y, node_pos.z, color='black', s=20)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Room Geometry')  # Simplified base title
    
    # Set axis limits
    ax.set_xlim([0, room.x])
    ax.set_ylim([0, room.y])
    ax.set_zlim([0, room.z])
    
    # Add legend
    ax.legend()
    
    # Make the plot interactive
    # plt.ion()  # Turn on interactive mode
    
    # Enable mouse rotation
    ax.mouse_init()
    
    return ax  # Return the axis for further plotting

def plot_ism_path(room, ism_calc, path: List[str], ax=None):
    """
    Visualize an ISM path with all image sources and reflection points.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plot_room(room, ax)
    
    # Get path without 'm' for dictionary lookups
    path_key = tuple(path[:-1])
    
    # Check if path exists in calculated paths
    if path_key not in ism_calc.node_positions:
        print(f"Warning: Path {' → '.join(path)} not found in calculated paths.")
        print(f"Make sure to calculate paths up to order {len(path)-2}")
        return ax
    
    # Set title to show the path
    path_str = ' → '.join(path)  # Using arrow instead of hyphen
    ax.set_title(f'ISM Path: {path_str}', pad=20)  # Added some padding
    
    # Calculate and collect all intermediate image sources
    image_sources = []
    for i in range(1, len(path)-1):  # -1 to exclude 'm'
        intermediate_path = path[:i+1]  # Include one more wall each time
        intermediate_key = tuple(intermediate_path)
        if intermediate_key not in ism_calc.image_sources:
            # Calculate if not already cached
            print("this print should never be necessary")
            image_source = ism_calc.calculate_image_source(intermediate_path)
        else:
            image_source = ism_calc.image_sources[intermediate_key]
        image_sources.append(image_source)
    
    # Get reflection points
    reflection_points = ism_calc.node_positions[path_key]
    
    # Plot image sources with different shades of purple
    purple_shades = plt.cm.Purples(np.linspace(0.5, 0.9, len(image_sources)))
    for i, source in enumerate(image_sources):
        ax.scatter(source.x, source.y, source.z,
                  color=purple_shades[i], s=100,
                  label=f'Image Source {i+1} ({path[i+1]} wall)')
    
    # Plot reflection points with colors matching their path segments
    reflection_colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(reflection_points)))
    for i, point in enumerate(reflection_points):
        ax.scatter(point.x, point.y, point.z,
                  color=reflection_colors[i], s=100,
                  label=f'Reflection Point {i+1} ({path[i+1]} wall)')
    
    # Get the path object from shared tracker
    path_obj = next(p for p in ism_calc.path_tracker.paths['ISM'][len(path)-2] 
                   if p.nodes == path)
    
    # Draw path segments with decreasing intensity and show lengths
    points = [room.source.srcPos] + reflection_points + [room.micPos]
    segment_colors = plt.cm.Reds(np.linspace(0.3, 0.7, len(points)-1))
    linewidths = np.linspace(3, 1, len(points)-1)
    alphas = np.linspace(1.0, 0.6, len(points)-1)
    
    # Plot segments with length labels
    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        
        # Draw the line segment
        ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z],
                '--',
                color=segment_colors[i],
                linewidth=linewidths[i],
                alpha=alphas[i],
                label=f'Path Segment {i+1}' if i==0 else None)
        
        # Calculate segment length
        segment_length = p1.getDistance(p2)
        
        # Calculate midpoint for label position
        mid_x = (p1.x + p2.x) / 2
        mid_y = (p1.y + p2.y) / 2
        mid_z = (p1.z + p2.z) / 2
        
        # Add length label with small offset
        offset = 0.1  # Adjust this value to change label position
        ax.text(mid_x + offset, mid_y + offset, mid_z + offset, 
                f'{segment_length:.2f}m',
                color=segment_colors[i],
                alpha=alphas[i])
    
    # Print path details using stored information
    print(f"\nPath: {' → '.join(path)}")
    print(f"Image source total distance: {path_obj.distance:.2f}m")
    print(f"Segment distances: {' + '.join(f'{d:.2f}' for d in path_obj.segment_distances)}m")
    print(f"Sum of segments distance: {path_obj.total_segment_distance:.2f}m")
    print(f"Difference: {abs(path_obj.distance - path_obj.total_segment_distance):.6f}m")
    if not path_obj.is_valid:
        print("(INVALID PATH)")
    
    plt.legend()
    return ax

def calculate_rt60_theoretical(room_dim, absorption):
    """Calculate theoretical RT60 using Sabine and Eyring formulas.
    
    Args:
        room_dim: Room dimensions [width, depth, height] in meters
        absorption: Average absorption coefficient
    
    Returns:
        rt60_sabine: Reverberation time using Sabine's formula
        rt60_eyring: Reverberation time using Eyring's formula
    """
    # Room volume and surface area
    V = room_dim[0] * room_dim[1] * room_dim[2]  # Volume
    S = 2 * (room_dim[0]*room_dim[1] + room_dim[1]*room_dim[2] + room_dim[0]*room_dim[2])  # Surface area
    
    # Sabine's formula
    rt60_sabine = 0.161 * V / (S * absorption)
    
    # Eyring's formula
    rt60_eyring = 0.161 * V / (-S * np.log(1 - absorption))
    
    return rt60_sabine, rt60_eyring

def calculate_rt60_from_rir(rir, fs):
    """Calculate RT60 from RIR using pyroomacoustics.
    
    Args:
        rir: Room impulse response
        fs: Sampling frequency
    
    Returns:
        rt60: Estimated RT60 value
    """
    # Normalize RIR
    rir = rir / np.max(np.abs(rir))

    # Estimate RT60
    rt60 = pra.experimental.rt60.measure_rt60(rir, fs)
    return rt60


