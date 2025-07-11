import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import pyroomacoustics as pra

from plotting_utils import DISPLAY_NAME_MAP, get_display_name, get_color

def plot_room(room, ax=None):
    """Plot room geometry with source, mic, and walls."""
    if ax is None:
        plt.close('all')  # Close any existing plots
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot source and microphone
    ax.scatter(room.source.sx, room.source.sy, room.source.sz, 
              color='green', s=100, label='Source')
    ax.scatter(room.micPos.x, room.micPos.y, room.micPos.z, 
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
            
        surf = ax.plot_surface(xx, yy, zz, alpha=0.3,
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

def calculate_rt60_from_rir(rir, fs, plot):
    # return error with message if rir is empty
    # new function is moved to analysis.py with the same name
    raise ValueError("this function is moved to analysis.py. Apparently this reference is forgotten to be updated. [ege]")

def create_interactive_rir_plot(rirs_dict, Fs, put_rect=True):
    """Create an interactive RIR plot with checkboxes to show/hide different RIRs.

    Args:
        rirs_dict: Dictionary containing RIRs with their labels as keys
        Fs: Sampling frequency
        put_rect: If True, adds a rectangle to highlight early reflections.
    """
    from matplotlib.widgets import CheckButtons

    # Create the main figure and axis for RIR plot
    fig, (ax, ax_check) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]}, figsize=(13, 6))

    # Initialize lines dictionary and visibility states
    lines = {}
    visibility = {}
    programmatic_to_display = {
        label: get_display_name(label.split(':')[0].strip(), {}, DISPLAY_NAME_MAP) 
        for label in rirs_dict.keys()
    }
    display_to_programmatic = {v: k for k, v in programmatic_to_display.items()}

    # colors = ["tab:blue", "red", "orange", "cyan"]

    # c = 0
    # alp = 0.7
    # Plot each RIR
    for i, (label, rir) in enumerate(rirs_dict.items()):
        # if c != 0:
        #     alp = 1 # decrease the alpha if you want the second plot more transparent
        # else:
        #     alp = 1
        time_axis = np.arange(len(rir)) / Fs
        display_label = programmatic_to_display[label]
        
        # Get color from central styling map
        method_key = label.split(':')[0].strip()
        color = get_color(method_key, DISPLAY_NAME_MAP)

        line, = ax.plot(time_axis, rir, label=display_label, visible=True, color=color)
        lines[label] = line
        visibility[label] = True
        # c += 1
        # alp *= 0.7

    # Set up the main plot
    # ax.set_title('Room Impulse Response Comparison')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    
    if put_rect:
        import matplotlib.patches as patches
        ymin, ymax = ax.get_ylim()
        ax.add_patch(
            patches.Rectangle(
                (0.017, 0.1), # x axis start, ymin start
                0.03, # 50 ms for early reflections
                0.2,
                facecolor="yellow",
                alpha=0.3,
                linewidth=0,
            ))

        ax.add_patch(
        patches.Rectangle(
            (0.0025, 0.88),  # x axis start, ymin
            0.038,  # 50 ms for early reflections
            0.15,
            facecolor="lime",
            alpha=0.3,
            linewidth=1,
        )
        )
    
    # Set up check buttons for toggling visibility
    labels = list(programmatic_to_display.values())
    actives = [True] * len(labels)

    # Remove the outer box and title from the checkbox area
    ax_check.set_xticks([])  # Hide x-ticks
    ax_check.set_yticks([])  # Hide y-ticks
    for spine in ax_check.spines.values():  # Remove spines (outer box)
        spine.set_visible(False)

    # Position the checkboxes in a good spot
    ax_check.set_position([0.8, 0.4, 0.1, 0.2])  # [left, bottom, width, height]
    check = CheckButtons(
        ax=ax_check,
        labels=labels,
        actives=actives
    )

    def update_visibility(display_label):
        # Toggle visibility of the corresponding line
        prog_label = display_to_programmatic[display_label]
        line = lines[prog_label]
        line.set_visible(not line.get_visible())
        fig.canvas.draw_idle()  # Redraw the figure

        # Update legend
        handles = [line for line in lines.values() if line.get_visible()]
        labels = [line.get_label() for line in lines.values() if line.get_visible()]
        ax.legend(handles, labels)


    # Connect the callback
    check.on_clicked(update_visibility)

    # Add initial legend
    handles = [line for line in lines.values() if line.get_visible()]
    labels = [line.get_label() for line in lines.values() if line.get_visible()]
    ax.legend(handles, labels)

    # Keep a reference to prevent garbage collection
    fig.check = check

    # plt.show(block=True)  # Make sure to block to keep the window interactive
    plt.show(block=False)  # Non-blocking

def create_interactive_edc_plot(rirs_dict, Fs):
    """Create an interactive EDC plot with checkboxes to show/hide different EDCs.
    
    Args:
        rirs_dict: Dictionary containing RIRs with their labels as keys
        Fs: Sampling frequency
    """
    from matplotlib.widgets import CheckButtons
    import analysis as an

    # Create the main figure and axis for EDC plot with more space for the plot
    fig, (ax, ax_check) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]}, figsize=(13, 6))

    # Initialize lines dictionary and visibility states
    lines = {}
    visibility = {}

    # Calculate and plot all EDCs initially
    for label, rir in rirs_dict.items():
        # Calculate EDC without plotting
        edc = an.compute_edc(rir, Fs, label=label, plot=False)
        
        # Create time array in seconds
        time = np.arange(len(rir)) / Fs
        
        # Determine color based on whether it's a default RIR
        color = None
        # Plot EDC
        line, = ax.plot(time, edc, label=label, alpha=1, color=color)
        lines[label] = line
        visibility[label] = True

    # Set up the main plot
    ax.set_title('Energy Decay Curve Comparison')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (dB)')
    ax.grid(True)
    ax.legend()
    ax.set_ylim(-65, 5)  # Set y-axis limits similar to the original EDC plot

    # Create CheckButtons
    labels = list(rirs_dict.keys())
    actives = [True] * len(labels)

    # Remove the outer box and title from the checkbox area
    ax_check.set_xticks([])  # Hide x-ticks
    ax_check.set_yticks([])  # Hide y-ticks
    for spine in ax_check.spines.values():  # Remove spines (outer box)
        spine.set_visible(False)

    # Position the checkboxes in a more compact spot
    ax_check.set_position([0.8, 0.4, 0.1, 0.2])  # [left, bottom, width, height]
    check = CheckButtons(
        ax=ax_check,
        labels=labels,
        actives=actives
    )

    def update_visibility(label):
        # Toggle visibility of the corresponding line
        line = lines[label]
        line.set_visible(not line.get_visible())
        fig.canvas.draw_idle()  # Redraw the figure

        # Update legend
        handles = [line for line in lines.values() if line.get_visible()]
        labels = [line.get_label() for line in lines.values() if line.get_visible()]
        ax.legend(handles, labels)

    # Connect the callback
    check.on_clicked(update_visibility)

    # Keep a reference to prevent garbage collection
    fig.check = check

    plt.show(block=False)  # Non-blocking

def create_unified_interactive_plot(rirs_dict, Fs, room_parameters=None, reflection_times=None):
    """Create a unified interactive plot with RIR and EDC side by side, and NED below, sharing synchronized checkboxes.

    Args:
        rirs_dict: Dictionary containing RIRs with their labels as keys
        Fs: Sampling frequency
        room_parameters: Dictionary containing room parameters including dimensions and display_name
        reflection_times: Dictionary containing arrival times for different reflection orders
    """
    from matplotlib.widgets import CheckButtons
    import analysis as an
    import EchoDensity as ned

    num_rirs_total = len(rirs_dict)
    initial_legend_fontsize = None
    if num_rirs_total <= 5:
        initial_legend_ncol = num_rirs_total if num_rirs_total > 0 else 1
    elif num_rirs_total <= 10:
        initial_legend_ncol = 5
    elif num_rirs_total <= 18: # max 3 rows for up to 18 items with 6 cols
        initial_legend_ncol = 6
    else: # many items
        initial_legend_ncol = 8 # try to fit more horizontally

    if num_rirs_total > 16:
        initial_legend_fontsize = 'small'

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    # Adjust constrained_layout padding
    fig.set_constrained_layout_pads(w_pad=2/72, h_pad=2/72, hspace=0.02, wspace=0.02)

    # Adjusted GridSpec: 3 rows (NED/EDC, Legend, RIR), 3 columns (plot, plot, checkboxes)
    # Increased height for legend row
    gs = plt.GridSpec(3, 3, height_ratios=[0.40, 0.18, 0.42], width_ratios=[1, 1, 0.2])
    
    ax_ned = fig.add_subplot(gs[0, 0])
    ax_edc = fig.add_subplot(gs[0, 1])
    # Legend container in the middle row, spanning plot columns
    ax_legend_container = fig.add_subplot(gs[1, 0:2])
    ax_legend_container.axis('off') # Hide axes for the legend container
    # RIR plot in the bottom row, spanning plot columns
    ax_rir = fig.add_subplot(gs[2, 0:2])
    # Checkboxes span all rows in the third column
    ax_check = fig.add_subplot(gs[:, 2])

    # Add room information at the top
    if room_parameters is not None:
        rp = room_parameters
        room_name = room_parameters.get('display_name', 'Custom Room')
        src = rp["source x"], rp["source y"], rp["source z"]
        mic = rp["mic x"], rp["mic y"], rp["mic z"]
        room_info = f"**{room_name}: {rp['width']}×{rp['depth']}×{rp['height']}m \
                    Source: {src[0]:.2f}m, {src[1]:.2f}m, {src[2]:.2f}m \
                    Microphone: {mic[0]:.2f}m, {mic[1]:.2f}m, {mic[2]:.2f}m  **"
        fig.suptitle(room_info, fontsize=12)

    # Initialize lines dictionaries for all plots
    rir_lines = {}
    edc_lines = {}
    ned_lines = {}
    visibility = {}

    # Plot all RIRs, EDCs, and NEDs initially
    for label, rir in rirs_dict.items():
        # Create time arrays in seconds
        time = np.arange(len(rir)) / Fs
        
        # Plot RIR
        rir_line, = ax_rir.plot(time, rir, label=label, alpha=0.7)
        rir_lines[label] = rir_line
        
        # Calculate and plot EDC
        edc, time, _ = an.compute_edc(rir, Fs, label=label, plot=False)
        
        edc_line, = ax_edc.plot(time, edc, label=label, alpha=0.7)
        edc_lines[label] = edc_line
        
        # Calculate and plot NED
        echo_density = ned.echoDensityProfile(rir, fs=Fs)
        ned_time = np.arange(len(echo_density)) / Fs
        ned_line, = ax_ned.plot(ned_time, echo_density, label=label, alpha=0.7)
        ned_lines[label] = ned_line
        
        visibility[label] = True

    # Add vertical lines for reflection arrival times if provided
    if reflection_times is not None:
        colors = ['red', 'blue', 'green']  # Colors for 1st, 2nd, and 3rd order reflections
        order_map = {
            'first_order': 1,
            'second_order': 2,
            'third_order': 3
        }
        for order_name, time in reflection_times.items():
            if time is not None:  # Only plot if time exists
                order_num = order_map.get(order_name)
                if order_num is not None:
                    time_sec = time  # time is already in seconds
                    color = colors[order_num - 1]
                    
                    # Add vertical lines to RIR plot (without label)
                    ax_rir.axvline(x=time_sec, color=color, linestyle='--', alpha=0.3)
                    # Add text annotation
                    ax_rir.text(time_sec - 0.02, 0.95, f'N={order_num}',
                              color=color, alpha=0.3, transform=ax_rir.get_xaxis_transform(),
                              verticalalignment='top')
                    
                    # Add vertical lines to EDC plot (without label)
                    ax_edc.axvline(x=time_sec, color=color, linestyle='--', alpha=0.3)
                    # Add text annotation
                    ax_edc.text(time_sec - 0.02, 0.95, f'N={order_num}',
                              color=color, alpha=0.3, transform=ax_edc.get_xaxis_transform(),
                              verticalalignment='top')

                    # Add vertical lines to NED plot (without label)
                    ax_ned.axvline(x=time_sec, color=color, linestyle='--', alpha=0.3)
                    # Add text annotation
                    ax_ned.text(time_sec - 0.02, 0.95, f'N={order_num}',
                              color=color, alpha=0.3, transform=ax_ned.get_xaxis_transform(),
                              verticalalignment='top')

    # Set up the RIR plot
    ax_rir.set_title('Room Impulse Response Comparison')
    ax_rir.set_xlabel('Time (s)')
    ax_rir.set_ylabel('Amplitude')
    ax_rir.grid(True)
    # ax_rir.legend() # Removed individual legend

    # Set up the EDC plot
    ax_edc.set_title('Energy Decay Curve Comparison')
    ax_edc.set_xlabel('Time (s)')
    ax_edc.set_ylabel('Energy (dB)')
    ax_edc.grid(True)
    # ax_edc.legend(loc='upper right', ncol=initial_legend_ncol, fontsize=initial_legend_fontsize) # Removed
    ax_edc.set_ylim(-65, 5)

    # Set up the NED plot
    ax_ned.set_title('Normalized Echo Density')
    ax_ned.set_xlabel('Time (s)')
    ax_ned.set_ylabel('Normalized Echo Density')
    ax_ned.grid(True)
    # ax_ned.legend(loc='upper right', ncol=initial_legend_ncol, fontsize=initial_legend_fontsize) # Removed
    # ax_ned.set_xscale('log')
    #ax_ned.set_xlim(left=100/Fs)  # Convert sample index to time

    # Create shared legend
    all_handles = [rir_lines[label] for label in rirs_dict.keys() if label in rir_lines and rir_lines[label].get_visible()]
    all_labels = [label for label in rirs_dict.keys() if label in rir_lines and rir_lines[label].get_visible()]
    if all_handles:
        fig.shared_legend = ax_legend_container.legend(
            all_handles, all_labels, 
            loc='center', 
            ncol=initial_legend_ncol, 
            fontsize=initial_legend_fontsize,
            frameon=False
        )

    # Create CheckButtons
    labels = list(rirs_dict.keys())
    actives = [True] * len(labels)

    # Remove the outer box and title from the checkbox area
    ax_check.set_xticks([])
    ax_check.set_yticks([])
    for spine in ax_check.spines.values():
        spine.set_visible(False)

    # Position the checkboxes
    # ax_check.set_position([0.85, 0.4, 0.1, 0.2]) # Commented out to let GridSpec and tight_layout manage
    check = CheckButtons(
        ax=ax_check,
        labels=labels,
        actives=actives
    )

    def update_visibility(label):
        # Toggle visibility of the corresponding lines in all plots
        rir_line = rir_lines[label]
        edc_line = edc_lines[label]
        ned_line = ned_lines[label]
        
        rir_line.set_visible(not rir_line.get_visible())
        edc_line.set_visible(not edc_line.get_visible())
        ned_line.set_visible(not ned_line.get_visible())
        
        num_visible_labels = sum(1 for line in rir_lines.values() if line.get_visible())
        
        updated_legend_fontsize = None
        if num_visible_labels <= 5:
            updated_legend_ncol = num_visible_labels if num_visible_labels > 0 else 1
        elif num_visible_labels <= 10:
            updated_legend_ncol = 5 
        elif num_visible_labels <= 18:
            updated_legend_ncol = 6
        else: # > 18
            updated_legend_ncol = 8

        if num_visible_labels > 16:
            updated_legend_fontsize = 'small'

        # Remove old shared legend if it exists
        if hasattr(fig, 'shared_legend') and fig.shared_legend:
            fig.shared_legend.remove()
            fig.shared_legend = None
        
        # Update shared legend
        visible_handles = [line for line in rir_lines.values() if line.get_visible()]
        visible_labels = [line.get_label() for line in rir_lines.values() if line.get_visible()]
        
        if visible_handles: # Only create legend if there are visible items
            fig.shared_legend = ax_legend_container.legend(
                visible_handles, visible_labels, 
                loc='center', 
                ncol=updated_legend_ncol, 
                fontsize=updated_legend_fontsize,
                frameon=False
            )
        
        fig.canvas.draw_idle()

    # Connect the callback
    check.on_clicked(update_visibility)

    # Keep a reference to prevent garbage collection
    fig.check = check

    plt.show(block=False)
