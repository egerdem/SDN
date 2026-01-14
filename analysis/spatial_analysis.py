import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import geometry
from sdn_core import DelayNetwork
from archive.sdn_base import calculate_sdn_base_rir
import pyroomacoustics as pra
from scipy import signal
# import seaborn as sns
import analysis as an
import rir_calculators as rir_calc
# Handle both relative import (when used as module) and absolute import (when run directly or from same dir)
try:
    from .plotting_utils import DISPLAY_NAME_MAP
except ImportError:
    from plotting_utils import DISPLAY_NAME_MAP


def print_receiver_grid(receiver_positions, room, source_position=None, save=False, source_name=""):
    if source_position is None:
        source_position = room["source x"], room["source y"], room["source z"]

    # Print and visualize the receiver grid
    print("\nReceiver Grid Coordinates:")
    grid_size = int(np.sqrt(len(receiver_positions)))

    for i, pos in enumerate(receiver_positions):
        rx, ry = pos[:2]
        print(f"Position {i:2d}: ({rx:.2f}, {ry:.2f}), grid index: row={i // grid_size}, col={i % grid_size}")

    # Visualize the receiver grid
    plt.figure(figsize=(10, 8))
    rx_values = [pos[0] for pos in receiver_positions]
    ry_values = [pos[1] for pos in receiver_positions]
    plt.scatter(rx_values, ry_values, c='blue', s=100, alpha=0.7)

    # Add position indices
    for i, pos in enumerate(receiver_positions):
        rx, ry = pos[:2]
        plt.text(rx, ry, f"{i}", fontsize=9, ha='center', va='center')

    # Add grid indices
    for i, pos in enumerate(receiver_positions):
        rx, ry = pos[:2]
        row, col = i // grid_size, i % grid_size
        # plt.text(rx, ry+0.2, f"({row},{col})", fontsize=8, ha='center', va='center', color='red')
        plt.text(rx, ry + 0.2, f"{rx:.2f}, {ry:.2f}", fontsize=8, ha='center', va='center', color='black')

    # Add source position
    plt.plot(source_position[0], source_position[1], 'ro', markersize=15)
    
    # Add source coordinates text above the source circle
    source_x, source_y, source_z = source_position[0], source_position[1], source_position[2]
    plt.text(source_x, source_y + 0.3, f'({source_x:.2f}, {source_y:.2f}, {source_z:.2f})', 
             fontsize=10, ha='center', va='bottom', color='red', weight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

    # Add room boundaries
    plt.plot([0, room['width']], [0, 0], 'k-', linewidth=2)
    plt.plot([0, room['width']], [room['depth'], room['depth']], 'k-', linewidth=2)
    plt.plot([0, 0], [0, room['depth']], 'k-', linewidth=2)
    plt.plot([room['width'], room['width']], [0, room['depth']], 'k-', linewidth=2)

    plt.title('Receiver Grid Positions')
    plt.xlabel('Room Width (m)')
    plt.ylabel('Room Depth (m)')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    name = room['display_name']

    if save:
        plt.savefig(f'{name}_{source_name}_receiver_grid_old_positions.png')
        plt.close()
    plt.show()

def print_source_grid(source_positions, receiver_positions, room, save=False, grid_name=""):
    """
    Visualize all source positions and receiver positions together in a single plot.
    
    Args:
        source_positions: List of source positions (x, y, z) or (x, y, z, name)
        receiver_positions: List of receiver positions (x, y, z)
        room: Room parameters dictionary
        save: Whether to save the plot
        grid_name: Name for the saved file
    """
    print(f"\nSource-Receiver Grid Visualization:")
    print(f"Sources: {len(source_positions)} positions")
    print(f"Receivers: {len(receiver_positions)} positions")
    
    # Print source coordinates
    print("\nSource Positions:")
    for i, pos in enumerate(source_positions):
        if len(pos) >= 4:  # Has name
            sx, sy, sz, name = pos[:4]
            print(f"Source {i:2d}: ({sx:.2f}, {sy:.2f}, {sz:.2f}) - {name}")
        else:
            sx, sy, sz = pos[:3]
            print(f"Source {i:2d}: ({sx:.2f}, {sy:.2f}, {sz:.2f})")
    
    # Print receiver coordinates
    print("\nReceiver Positions:")
    grid_size = int(np.sqrt(len(receiver_positions)))
    for i, pos in enumerate(receiver_positions):
        rx, ry = pos[:2]
        print(f"Receiver {i:2d}: ({rx:.2f}, {ry:.2f}), grid index: row={i // grid_size}, col={i % grid_size}")

    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot receivers
    rx_values = [pos[0] for pos in receiver_positions]
    ry_values = [pos[1] for pos in receiver_positions]
    plt.scatter(rx_values, ry_values, c='blue', s=80, alpha=0.7, marker='o', label='Receivers')

    # Add receiver indices and coordinates
    for i, pos in enumerate(receiver_positions):
        rx, ry = pos[:2]
        plt.text(rx, ry, f"R{i}", fontsize=8, ha='center', va='center', color='white', weight='bold')
        plt.text(rx, ry + 0.15, f"({rx:.1f},{ry:.1f})", fontsize=7, ha='center', va='center', color='blue')

    # Plot sources
    sx_values = [pos[0] for pos in source_positions]
    sy_values = [pos[1] for pos in source_positions]
    plt.scatter(sx_values, sy_values, c='red', s=120, alpha=0.8, marker='*', label='Sources')
    
    # Add source indices and coordinates
    for i, pos in enumerate(source_positions):
        sx, sy = pos[:2]
        sz = pos[2] if len(pos) > 2 else 0
        plt.text(sx, sy, f"S{i}", fontsize=9, ha='center', va='center', color='white', weight='bold')
        
        # Add source name if available
        if len(pos) >= 4:
            name = pos[3]
            plt.text(sx, sy - 0.25, f"{name}", fontsize=8, ha='center', va='top', color='red', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.8))
        else:
            plt.text(sx, sy - 0.25, f"({sx:.1f},{sy:.1f},{sz:.1f})", fontsize=8, ha='center', va='top', color='red')

    # Add room boundaries
    plt.plot([0, room['width']], [0, 0], 'k-', linewidth=3, label='Room boundaries')
    plt.plot([0, room['width']], [room['depth'], room['depth']], 'k-', linewidth=3)
    plt.plot([0, 0], [0, room['depth']], 'k-', linewidth=3)
    plt.plot([room['width'], room['width']], [0, room['depth']], 'k-', linewidth=3)

    plt.title(f'Combined Source-Receiver Grid Layout\n{room["display_name"]} - {len(source_positions)} Sources, {len(receiver_positions)} Receivers')
    plt.xlabel('Room Width (m)')
    plt.ylabel('Room Depth (m)')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    
    # Add room dimensions as text
    plt.text(room['width']/2, -0.3, f"Width: {room['width']}m", ha='center', va='top', fontsize=10)
    plt.text(-0.3, room['depth']/2, f"Depth: {room['depth']}m", ha='center', va='center', rotation=90, fontsize=10)

    if save:
        # If grid_name is an absolute path, use it directly; otherwise construct filename
        if os.path.isabs(grid_name):
            filename = grid_name
        else:
            filename = f'{room["display_name"]}_{grid_name}_combined_source_receiver_grid.png'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        plt.close()
    else:
        plt.show()

def generate_receiver_grid_old(room_width: float, room_depth: float, room_height: float = None, z: float = None, margin = 1, center_margin = None, n_points: int = 50) -> List[Tuple[float, float, float]]:
    """Generate a grid of receiver positions within the room.
    
    Args:
        room_width (float): Width of the room
        room_depth (float): Depth of the room
        room_height (float): Height of the room (optional, used to determine default z)
        z (float): Height of receivers. If None, automatically determined from room dimensions
        margin (float): Wall margin (default: 1)
        center_margin (float): Center margin, if None uses margin value
        n_points (int): Number of receiver positions to generate
        
    Returns:
        List[Tuple[float, float, float]]: List of (x, y, z) coordinates for receivers
    """
    # Auto-determine z height based on room dimensions if not provided
    if z is None:
        # Match against known room configurations from experiment_configs
        if abs(room_width - 9) < 0.01 and abs(room_depth - 7) < 0.01:  # AES Room
            z = 1.5
        elif abs(room_width - 6) < 0.01 and abs(room_depth - 7) < 0.01:  # WASPAA Room
            z = 2.4
        elif abs(room_width - 3.2) < 0.01 and abs(room_depth - 4) < 0.01:  # Journal Room
            z = 1.5
        elif abs(room_width - 6) < 0.01 and abs(room_depth - 6) < 0.01:  # Cube6 Room
            z = 3.0
        else:
            # Default fallback
            assert room_height is not None, "Either z or room_height must be provided"
    
    # Create a grid of points, avoiding walls (1m margin)
    # margin is WALL MARGIN
    if center_margin is None:
        center_margin = margin
    x = np.linspace(margin, room_width - center_margin, int(np.sqrt(n_points)))
    y = np.linspace(margin, room_depth - center_margin, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)
    return list(zip(X.flatten(), Y.flatten(), [z] * len(X.flatten())))


def generate_receiver_grid_tr(room_width: float, room_depth: float, margin=1, n_points: int = 50) -> List[
    Tuple[float, float]]:
    """Generate a grid of receiver positions within the room.

    Args:
        room_width (float): Width of the room
        room_depth (float): Depth of the room
        n_points (int): Number of receiver positions to generate

    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinates for receivers
    """
    # Create a grid of points, avoiding walls with margin
    margin_from_center = margin - 0.2 # less margin for center
    x = np.linspace(margin, room_width - margin_from_center, int(np.sqrt(n_points)))
    y = np.linspace(margin, room_depth - margin_from_center, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)
    print(f"Receiver grid: {X.shape} positions, margin={margin}, center margin={margin_from_center}")
    return list(zip(X.flatten(), Y.flatten()))

def generate_full_receiver_grid(room_width: float, room_depth: float, height: float, n_x: int = 4, n_y: int = 4, margin: float = 0.5) -> List[Tuple[float, float, float]]:
    """
    Generate a grid of receiver positions covering the entire room.
    
    Args:
        room_width: Width of room
        room_depth: Depth of room
        height: Height of receivers
        n_x: Number of points along width
        n_y: Number of points along depth
        margin: Distance from walls
    """
    x = np.linspace(margin, room_width - margin, n_x)
    y = np.linspace(margin, room_depth - margin, n_y)
    
    receivers = []
    for xi in x:
        for yi in y:
            receivers.append((xi, yi, height))
            
    return receivers

def generate_source_positions(room_params, name = None):
    """Create a list of source positions within the room.

    Args:
        room_params (dict): Room parameters including dimensions and source positions.
            Must contain 'width', 'depth', and 'source z' keys.

    Returns:
        If method is "sdn": List of (x, y, z) position tuples
    """
    if name is None:
        assert False, "Source position version name must be specified (e.g., 'v1' or 'v2')"

    # Extract room dimensions
    room_width = room_params['width']
    room_depth = room_params['depth']
    room_height = room_params['height']
    source_z = room_params['source z']

    # Define source positions
    if name == "v1":
        source_positions = [
            # Source in the middle of the room
            (room_width / 2, room_depth / 2, source_z, "Center_Source"),

            # Source in the lower left corner
            (1.0, 1.0, source_z, "Lower_Left_Source"),

            # Source in the upper right corner, offset from the right wall
            (room_width - 0.5, room_depth - 1.0, source_z, "Upper_Right_Source"),

            # Source at the top middle wall
            (room_width / 2, room_depth - 1.0, source_z, "Top_Middle_Source")
        ]
    elif name == "v2":
        source_positions = [
            # Source in the middle of the room
            (room_width / 2, room_depth / 2, source_z, "Center_Source"),

            # Source in the lower left corner
            (1.0, 1.0, source_z, "Lower_Left_SourceV2"),

            # Source in the upper right corner, offset from the right wall
            (room_width - 1, room_depth - 1.0, source_z, "Upper_Right_SourceV2"),

            # Source at the top middle wall
            (room_width / 2, room_depth - 1.0, source_z, "Top_Middle_SourceV2")
        ]

    elif name == "v3":
        source_positions = [
            # Source in the middle of the room
            (room_width / 2, room_depth / 2, source_z, "Center_Source"),

            # Source in the lower left corner
            (1.0, 1.0, source_z, "Lower_Left_Source"),

            # Source in the upper right corner, offset from the right wall
            (room_width - 0.5, room_depth - 1.0, source_z, "Upper_Right_Source"),

            # Source at the top middle wall
            (room_width / 2, room_depth - 1.0, source_z, "Top_Middle_Source"),

            # corner source near the intersection of three walls
            (room_width - 0.5, room_depth - 1.0, room_height-0.5, "Corner_SourceV3")
        ]

    # return position tuples for SDN
    sources = [(pos[0], pos[1], pos[2], pos[3]) for pos in source_positions]

    return sources

def merge_source_lists(*source_lists):
    """Merge multiple source position lists, removing duplicates while preserving order.

    Args:
        *source_lists: Multiple lists of source positions (x, y, z) or (x, y, z, name)

    Returns:
        List of unique source positions
    """
    seen = set()
    merged_sources = []

    for source_list in source_lists:
        for src in source_list:
            key = (round(src[0], 6), round(src[1], 6), round(src[2], 6))
            if key not in seen:
                seen.add(key)
                merged_sources.append(src)

    return merged_sources


def generate_fixed_source_grid(
    room_params,
    padding,
    n_x=5,
    n_y=5,
    z_mode="mid",
    include_corners=True,
    corner_offset=1.5,
    diagonals=False
):
    """
    Deterministic, equi-spaced source positions (no random branching).

    Rationale: replaces the previous "corner_biased" RNG logic with a repeatable
    coverage pattern similar in spirit to your fixed sources, but denser/more representative.

    - n_x, n_y: grid density in x/y between [padding, dim-padding]. The 4 natural corner 
      positions from the grid are excluded, leaving (n_x * n_y - 4) grid sources.
    - z_mode:
        - "mid": z = h/2 (clipped to [padding, h-padding])
        - "fixed_1p5": z = 1.5 (clipped to [padding, h-padding])
    - include_corners: add 4 additional corner points at the same z using corner_offset
    - corner_offset: distance (meters) from each wall for the 4 corner points.
      This helps avoid exact overlap with receiver grids that often use `padding`
      as their margin (e.g., 0.5m). Default: 1.5m.
      
    Total sources: (n_x * n_y - 4) + (4 if include_corners else 0)
    Example: 5x5 grid -> 21 grid sources + 4 corners = 25 total sources
    """

    w = room_params['width']
    d = room_params['depth']
    h = room_params['height']

    # Generate grid points excluding corners (which will be added separately if include_corners=True)
    x_vals = np.linspace(padding, w - padding, n_x)
    y_vals = np.linspace(padding, d - padding, n_y)

    if z_mode == "fixed_1p5":
        z = float(np.clip(1.5, padding, h - padding))
    else:
        z = float(np.clip(h / 2.0, padding, h - padding))

    # Create all grid combinations but exclude the 4 corner positions
    all_grid_sources = [(float(x), float(y), z) for x in x_vals for y in y_vals]
    
    # Remove the 4 corner positions from the grid (first, last combinations)
    corner_positions = [
        (x_vals[0], y_vals[0]),   # bottom-left
        (x_vals[-1], y_vals[0]),  # bottom-right  
        (x_vals[0], y_vals[-1]),  # top-left
        (x_vals[-1], y_vals[-1])  # top-right
    ]
    
    sources = []
    for x, y, z_val in all_grid_sources:
        if (x, y) not in corner_positions:
            sources.append((x, y, z_val))

    if include_corners:
        # Place corner points *inside* the room, not exactly on the `padding` margin.
        # This reduces source/receiver "clashes" when receiver grids sit at `padding`.
        cx0 = float(np.clip(corner_offset, padding, w - padding))
        cy0 = float(np.clip(corner_offset, padding, d - padding))
        cx1 = float(np.clip(w - corner_offset, padding, w - padding))
        cy1 = float(np.clip(d - corner_offset, padding, d - padding))

        corners = [
            (cx0, cy0, z),
            (cx1, cy0, z),
            (cx0, cy1, z),
            (cx1, cy1, z),
        ]
        # Prepend corners so src1..src4 are always "boundary-like"
        sources = corners + sources

    # De-duplicate while preserving order
    seen = set()
    unique_sources = []
    for s in sources:
        key = (round(s[0], 6), round(s[1], 6), round(s[2], 6))
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    if diagonals:
        # Extract diagonal sources only (to get varying H_src values)
        # We'll filter to keep sources that lie approximately on the diagonal
        diag_sources = []

        # Create a grid to identify diagonal positions
        x_vals = np.linspace(0.5, w - 0.5, n_x)
        y_vals = np.linspace(0.5, d - 0.5, n_y)

        # For each source, check if it's on the diagonal
        for sx, sy, sz in unique_sources:
            # Find closest grid indices
            x_idx = np.argmin(np.abs(x_vals - sx))
            y_idx = np.argmin(np.abs(y_vals - sy))

            # Keep if on diagonal (x_idx == y_idx)
            if x_idx == y_idx and sx <= w/2 and sy <= d/2:
                diag_sources.append((sx, sy, sz))

        unique_sources = diag_sources[1:]

    return unique_sources


def generate_fixed_source_grid_simple(
        room_params,
        padding,
        n_x=2,
        n_y=2,
        z_mode="mid"):
    """
    Generate a simple 2x2 grid of sources positioned at padding distance from walls.
    
    Args:
        room_params: Room parameters dict with 'width', 'depth', 'height'
        padding: Distance from walls to place sources
        n_x: Number of sources along x-axis (default: 2)
        n_y: Number of sources along y-axis (default: 2) 
        z_mode: Height mode ("mid" for room center, "fixed_1p5" for 1.5m)
    
    Returns:
        List of (x, y, z, name) source positions
    """
    w = room_params['width']
    d = room_params['depth']
    h = room_params['height']

    # Calculate source positions at padding distance from walls
    x_vals = [padding, w - padding]  # Near left wall, near right wall
    y_vals = [padding, d - padding]  # Near front wall, near back wall

    if z_mode == "fixed_1p5":
        z = 1.5
    else:
        z = float(np.clip(h / 2.0, padding, h - padding))

    # Create 4 sources at the corners (padding distance from walls)
    sources = []
    source_names = ["Corner_BL", "Corner_TL", "Corner_BR", "Corner_TR"]  # Bottom-Left, Top-Left, Bottom-Right, Top-Right
    
    idx = 0
    for x in x_vals:
        for y in y_vals:
            # sources.append((float(x), float(y), z, source_names[idx]))
            sources.append((float(x), float(y), z))
            idx += 1

    return sources

def plot_rirs(rir_methods: Dict, receiver_positions: List[Tuple[float, float]],
             selected_positions: List[int] = None, Fs: int = 44100):
    """Plot RIRs from different methods at specified receiver positions.

    Args:
        rir_methods (Dict): Dictionary containing RIRs for each method
        receiver_positions (List[Tuple[float, float]]): List of receiver positions
        selected_positions (List[int], optional): Indices of positions to plot. If None, plots all.
        Fs (int): Sampling frequency for time axis
    """
    if selected_positions is None:
        # If no positions specified, plot first 4 positions
        selected_positions = list(range(min(4, len(receiver_positions))))

    n_positions = len(selected_positions)
    n_methods = len(rir_methods)

    fig, axes = plt.subplots(n_positions, 1, figsize=(12, 4*n_positions))
    if n_positions == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for pos_idx, ax in zip(selected_positions, axes):
        rx, ry = receiver_positions[pos_idx][:2]

        for method_idx, (method, rirs) in enumerate(rir_methods.items()):
            time = np.arange(len(rirs[pos_idx])) / Fs * 1000  # Convert to milliseconds
            ax.plot(time, rirs[pos_idx], label=method, color=colors[method_idx], alpha=0.7)

        ax.set_title(f'Receiver Position ({rx:.1f}m, {ry:.1f}m)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

def spatial_error_analysis(room_params: dict, source_pos: Tuple[float, float, float],
                           receiver_positions: List[Tuple[float, float]],
                         duration: float, Fs: int, err_duration_ms: float,
                         methods: List[str],
                         method_configs: Dict,
                         reference_method: str = 'ISM',
                         comparison_type: str = "edc",
                         error_metric: str = None) -> Dict:
    """Perform spatial error analysis between different RIR calculation methods.

    Args:
        room_params (dict): Room parameters
        source_pos (Tuple[float, float, float]): Source position
        duration (float): Duration of RIR
        Fs (int): Sampling frequency
        methods (List[str]): List of methods to compare
        method_configs (Dict): Configuration dictionary for each method
        reference_method (str): The method to use as the reference for comparison.
        comparison_type (str): Type of comparison to perform:
            - "smoothed_energy": Compare smoothed energy (default)
            - "energy": Compare raw energy
            - "edc": Compare Energy Decay Curves
        error_metric (str): Error metric to use:
            - "rmse": Root Mean Square Error (default)
            - "mae": Mean Absolute Error
            - "median": Median of absolute differences
            - "sum": Sum of absolute differences

    Returns:
        Dict: Dictionary containing error maps for each method comparison
    """
    # Generate receiver positions -- first version no margin
    # receiver_positions = generate_receiver_grid_tr(room_params['width'], room_params['depth'])

    # Initialize room and the source signal
    room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
    num_samples = int(Fs * duration)
    impulse = geometry.Source.generate_signal('dirac', num_samples)
    room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)

    # Calculate reflection coefficient
    room.wallAttenuation = [np.sqrt(1 - room_params['absorption'])] * 6
    room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])

    # Initialize error maps
    error_maps = {}
    rir_methods = {}

    # Setup grid for plotting
    grid_size = int(np.sqrt(len(receiver_positions)))
    X = np.array([pos[0] for pos in receiver_positions]).reshape(grid_size, grid_size)
    Y = np.array([pos[1] for pos in receiver_positions]).reshape(grid_size, grid_size)

    # Room dimensions for PRA
    room_dim = np.array([room_params['width'], room_params['depth'], room_params['height']])

    # Calculate RIRs for each method and position
    for method in methods:
        print(f"Calculating RIRs for method: {method}")
        print(f"Method config: {method_configs[method]}")
        rir_methods[method] = []
        config = method_configs[method]
        
        for pos in receiver_positions:
            rx, ry = pos[:2]
            room.set_microphone(rx, ry, room_params['mic z'])
            print(f"Receiver Position: ({rx:.2f}, {ry:.2f}")
            # This geom_room is needed for normalization later
            geom_room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
            geom_room.set_microphone(rx, ry, room_params['mic z'])
            geom_room.set_source(source_pos[0], source_pos[1], source_pos[2])

            if method == 'ISM':
                # Create a copy of room parameters with current mic position
                current_params = room_params.copy()
                current_params.update({
                    'mic x': rx,
                    'mic y': ry
                })
                max_order = config.get('max_order')
                # Calculate RIR using the unified function from rir_calculators
                rir, _ = rir_calc.calculate_pra_rir(current_params, duration, Fs, max_order)

            elif method == 'SDN-Base':
                # Create a copy of room parameters with current mic position
                current_params = room_params.copy()
                current_params.update({
                    'mic x': rx,
                    'mic y': ry
                })
                rir = calculate_sdn_base_rir(current_params, duration, Fs)

            elif method.startswith('SDN-'):
                # Handle all SDN methods (including SDN-Test and SDN-Original)
                _, rir, _, _ = rir_calc.calculate_sdn_rir(room_parameters, method, room, duration, Fs, config)
            else:
                print(f"Warning: Unknown method {method}, skipping")
                continue

            # Normalize RIR for all methods consistently, mirroring sdn_experiment_manager
            rir = rir_calc.rir_normalisation(rir, geom_room, Fs, normalize_to_first_impulse=True)['single_rir']

            rir_methods[method].append(rir)

    # Calculate error maps for each method comparison and print tables
    grid_size = int(np.sqrt(len(receiver_positions)))
    X = np.array([pos[0] for pos in receiver_positions]).reshape(grid_size, grid_size)
    Y = np.array([pos[1] for pos in receiver_positions]).reshape(grid_size, grid_size)
    error_maps = {}
    mean_errors = {}

    # Initialize error_maps structure
    for method in methods:
        if method != reference_method:
            comparison_key = f'{reference_method}_vs_{method}'
            error_maps[comparison_key] = {'X': X, 'Y': Y, 'errors': []}

    for pos_idx, pos in enumerate(receiver_positions):
        rx, ry = pos[:2]
        print(f"\n--- Results for Receiver Position {pos_idx}: ({rx:.2f}m, {ry:.2f}m) ---")
        table_data = []
        ref_rir = rir_methods[reference_method][pos_idx]

        for method in methods:
            rir = rir_methods[method][pos_idx]

            # Calculate metrics for the table
            rt60 = an.calculate_rt60_from_rir(rir, Fs, plot=False) if duration > 0.7 else 'N/A'
            _, energy, _ = an.calculate_err(rir, Fs=Fs)
            total_energy = np.sum(energy)

            # Calculate error for the specified duration
            err_samples = int(err_duration_ms / 1000 * Fs)
            if comparison_type == "smoothed_energy":
                sig1_full = an.calculate_smoothed_energy(ref_rir, window_length=30, Fs=Fs)
                sig2_full = an.calculate_smoothed_energy(rir, window_length=30, Fs=Fs)
                sig1, sig2 = sig1_full[:err_samples], sig2_full[:err_samples]
            elif comparison_type == "energy":
                sig1_full, _, _ = an.calculate_err(ref_rir, Fs=Fs)
                sig2_full, _, _ = an.calculate_err(rir, Fs=Fs)
                sig1, sig2 = sig1_full[:err_samples], sig2_full[:err_samples]
            elif comparison_type == "edc":
                sig1_full, _, _ = an.compute_edc(ref_rir, Fs, plot=False)
                sig2_full, _, _ = an.compute_edc(rir, Fs, plot=False)
                sig1, sig2 = sig1_full[:err_samples], sig2_full[:err_samples]
            else:
                raise ValueError(f"Unknown comparison type: {comparison_type}")

            if comparison_type == 'edc':
                error = an.compute_RMS(
                    sig1, 
                    sig2, 
                    range=int(err_duration_ms),
                    Fs=Fs,
                    skip_initial_zeros=False,
                    normalize_by_active_length=True
                )
            else: # rir
                error = an.compute_RMS(
                    sig1, 
                    sig2, 
                    range=int(err_duration_ms),
                    Fs=Fs
                )

            if method == reference_method:
                error = 0.0
            
            if method != reference_method:
                comparison_key = f'{reference_method}_vs_{method}'
                error_maps[comparison_key]['errors'].append(error)

            # Get display name from plotting_utils
            display_name = DISPLAY_NAME_MAP.get(method, method)
            table_data.append({
                'Method': display_name,
                'RT60': f"{rt60:.2f}" if isinstance(rt60, float) else rt60,
                'Total Energy': f"{total_energy:.6f}",
                f'RMSE [{err_duration_ms}ms]': f"{error:.6f}"
            })
        
        # Print the table for the current position
        header = f"{'Method':<30} {'RT60':>10} {'Total Energy':>18} {f'RMSE [{err_duration_ms}ms]':>18}"
        print(header)
        print("-" * len(header))
        for row in table_data:
            rmse_key = f'RMSE [{err_duration_ms}ms]'
            print(f"{row['Method']:<30} {row['RT60']:>10} {row['Total Energy']:>18} {row[rmse_key]:>18}")

    # Finalize error maps by reshaping the lists of errors into grids
    for key in error_maps:
        error_maps[key]['errors'] = np.array(error_maps[key]['errors']).reshape(grid_size, grid_size)
        mean_errors[key] = np.mean(error_maps[key]['errors'])

    # Print mean errors
    print("\nMean Errors Across All Positions:")
    print("-" * 80)
    print(f"{'Method Comparison':<30} {error_metric.upper():>15}{' ':>5}{'Info':>20}")
    print("-" * 80)
    for comparison, mean_error in mean_errors.items():
        # Extract method names from comparison key (e.g., 'ISM_vs_SDN-Test1')
        method1, method2 = comparison.split('_vs_')
        # Get info strings for both methods
        info1 = method_configs[method1].get('info', '')
        info2 = method_configs[method2].get('info', '')
        # Combine info strings
        combined_info = f"{info1} vs {info2}" if info1 and info2 else info1 or info2
        print(f"{comparison:<30} {mean_error:>15.6f}     {combined_info:>20}")
    print("-" * 80)
    
    return error_maps, rir_methods

def plot_error_maps(error_maps: Dict, room_params: dict, source_pos: Tuple[float, float, float], 
                   interpolated: bool = True, comparison_type: str = "edc", error_metric: str = "rmse", reference_method: str = 'ISM'):
    """Plot error maps with room layout.
    
    Args:
        error_maps (Dict): Dictionary containing error maps
        room_params (dict): Room parameters
        source_pos (Tuple[float, float, float]): Source position
        interpolated (bool): Whether to use interpolation in the plot (default: True)
        comparison_type (str): Type of comparison used
        error_metric (str): Error metric used
        reference_method (str): The method used as the reference for comparison.
    """
    # Filter for only reference method comparisons
    ref_comparisons = {k: v for k, v in error_maps.items() if k.startswith(f'{reference_method}_')}
    
    if not ref_comparisons:
        print(f"No comparisons with reference method '{reference_method}' found in error maps.")
        return
        
    n_comparisons = len(ref_comparisons)
    fig, axes = plt.subplots(n_comparisons, 1, figsize=(10, 5*n_comparisons))
    if n_comparisons == 1:
        axes = [axes]
    
    # Find global min and max for consistent colormap across subplots
    all_errors = np.concatenate([data['errors'].flatten() for data in ref_comparisons.values()])
    vmin, vmax = np.min(all_errors), np.max(all_errors)
    
    for ax, (comparison, data) in zip(axes, ref_comparisons.items()):
        # Plot error map
        if interpolated:
            # Interpolated contour plot
            contour = ax.contourf(data['X'], data['Y'], data['errors'], 
                                levels=20, cmap='viridis',
                                vmin=vmin, vmax=vmax)
        else:
            # Non-interpolated pcolormesh plot
            contour = ax.pcolormesh(data['X'], data['Y'], data['errors'],
                                  cmap='viridis', shading='nearest',
                                  vmin=vmin, vmax=vmax)
        
        # Plot receiver positions
        ax.scatter(data['X'].flatten(), data['Y'].flatten(), 
                  color='lightgray', alpha=0.4, s=30, 
                  marker='o', edgecolor='white', linewidth=0.5,
                  label='Receivers')
        
        # Plot room boundaries
        ax.plot([0, room_params['width']], [0, 0], 'k-', linewidth=2)
        ax.plot([0, room_params['width']], [room_params['depth'], room_params['depth']], 'k-', linewidth=2)
        ax.plot([0, 0], [0, room_params['depth']], 'k-', linewidth=2)
        ax.plot([room_params['width'], room_params['width']], [0, room_params['depth']], 'k-', linewidth=2)
        
        # Plot source position
        ax.plot(source_pos[0], source_pos[1], 'r*', markersize=15, label='Source')
        
        # Add colorbar and labels
        plt.colorbar(contour, ax=ax, label=f'{error_metric.upper()} Error')
        ax.set_title(f'Error Map: {comparison}\n({comparison_type} comparison, {error_metric} metric)')
        ax.set_xlabel('Room Width (m)')
        ax.set_ylabel('Room Depth (m)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    room_aes = {
        'display_name': 'AES Room',
        'width': 9, 'depth': 7, 'height': 4,
        'source x': 4.5, 'source y': 3.5, 'source z': 2,
        'mic z': 1.5,
        'absorption': 0.2,
        # 'air': {'humidity': 50,
        #        'temperature': 20,
        #        'pressure': 100},
    }

    room = room_aes
    room_parameters = room_aes
    ENABLED = False  # Set to True to enable all methods

    # Method configurations
    method_configs = {
        'ISM': {
            'enabled': ENABLED,
                'max_order': 100,
            'info': ''  # Empty info for ISM
        },
        'SDN-Test1': {
            'enabled': False,
                'label': "",
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': 1
                },
            'info': 'c1 original'
        },
        'SDN-Test5': {
            'enabled': ENABLED,
            'info': "c5",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 5,
            },
            'label': "SDN Test 5"
        },
        'SDN-Test6': {
            'enabled': ENABLED,
            'info': "c6",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 6,
            },
            'label': "SDN Test 6"
        },
        'SDN-Test7': {
            'enabled': ENABLED,
            'info': "c7",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 7,
            },
            'label': "SDN Test 7"
        }

    }

    # Get list of enabled methods
    enabled_methods = [method for method, config in method_configs.items() 
                      if config['enabled']]
    
    # Define the reference method for comparison
    reference_method = 'ISM'


    source_position = (room_parameters['source x'], 
                      room_parameters['source y'], 
                      room_parameters['source z'])

    # Analysis parameters
    USE_INTERPOLATION = False  # Set to False for discrete visualization
    
    # Example of different comparison types
    comparison_configs = [
        # {"type": "smoothed_energy", "metric": "median"},
        # {"type": "smoothed_energy", "metric": "sum"},
        {"type": "edc", "metric": "rmse"},
        # {"type": "edc", "metric": "sum"}
    ]

    receiver_positions = generate_receiver_grid_old(room['width'], room['depth'], room['height'], n_points=16,
                                                  margin=0.5)  # room aes

    print_receiver_grid(receiver_positions,room_parameters)

    # Run analysis for each configuration
    for config in comparison_configs:
        print(f"\nRunning analysis with {config['type']} comparison and {config['metric']} metric...")
        error_maps, rir_methods = spatial_error_analysis(
            room_parameters,
            source_position,
            receiver_positions,
            duration= 1,
            err_duration_ms=50,
            Fs=44100,
            methods=enabled_methods,
            method_configs=method_configs,  # Pass method configurations
            reference_method=reference_method,
            comparison_type=config['type'],
            error_metric=config['metric']
        )
        
        # Plot error maps with chosen visualization method
        plot_error_maps(error_maps, room_parameters, source_position, 
                       interpolated=USE_INTERPOLATION,
                       comparison_type=config['type'],
                       error_metric=config['metric'],
                       reference_method=reference_method)
        
        # Plot RIRs for first 4 receiver positions
        plot_rirs(rir_methods, receiver_positions, selected_positions=[0, 1, 2, 3])
