import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

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
        offset = 3.3  # Adjust this value to control label distance from wall
        
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
    plt.ion()  # Turn on interactive mode
    
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
    
    # Draw path segments with decreasing intensity
    points = [room.source.srcPos] + reflection_points + [room.micPos]
    segment_colors = plt.cm.Reds(np.linspace(0.3, 0.7, len(points)-1))
    linewidths = np.linspace(3, 1, len(points)-1)
    alphas = np.linspace(1.0, 0.6, len(points)-1)
    
    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z],
                '--',
                color=segment_colors[i],
                linewidth=linewidths[i],
                alpha=alphas[i],
                label=f'Path Segment {i+1}' if i==0 else None)
    
    # Calculate and print path segment distances
    points = [room.source.srcPos] + reflection_points + [room.micPos]
    segment_distances = []
    total_distance = 0
    
    print(f"\nPath: {' → '.join(path)}")
    print("Segment distances:")
    
    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        distance = p1.getDistance(p2)
        segment_distances.append(distance)
        total_distance += distance
        
        # Get segment description
        if i == 0:
            desc = "Source → First reflection"
        elif i == len(points)-2:
            desc = f"Reflection {i} → Mic"
        else:
            desc = f"Reflection {i} → Reflection {i+1}"
        
        print(f"  {desc}: {distance:.2f}m")
    
    # Print equation-style total
    equation = " + ".join([f"{d:.2f}" for d in segment_distances])
    print(f"Total distance: {equation} = {total_distance:.2f}m")
    
    plt.legend()
    return ax
