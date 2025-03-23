import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import geometry
from sdn_core import DelayNetwork
from sdn_base import calculate_sdn_base_rir
import pyroomacoustics as pra
from scipy import signal
import seaborn as sns
from analysis import calculate_smoothed_energy, calculate_error_metric, plot_smoothing_comparison, compute_RMS, compute_edc

def generate_receiver_grid(room_width: float, room_depth: float, n_points: int = 50) -> List[Tuple[float, float]]:
    """Generate a grid of receiver positions within the room.
    
    Args:
        room_width (float): Width of the room
        room_depth (float): Depth of the room
        n_points (int): Number of receiver positions to generate
        
    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinates for receivers
    """
    # Create a grid of points, avoiding walls (1m margin)
    margin = 1
    x = np.linspace(margin, room_width - margin, int(np.sqrt(n_points)))
    y = np.linspace(margin, room_depth - margin, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)
    return list(zip(X.flatten(), Y.flatten()))


def generate_source_positions(room_params: dict) -> list:
    """Create a list of source positions within the room.

    Args:
        room_params (dict): Room parameters including dimensions and source positions.
            Must contain 'width', 'depth', and 'source z' keys.

    Returns:
        If method is "sdn": List of (x, y, z) position tuples
    """
    # Extract room dimensions
    room_width = room_params['width']
    room_depth = room_params['depth']
    source_z = room_params['source z']

    # Define source positions
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

    # return position tuples for SDN
    sources = [(pos[0], pos[1], pos[2], pos[3]) for pos in source_positions]

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
        rx, ry = receiver_positions[pos_idx]

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
                         duration: float, Fs: int, methods: List[str], 
                         method_configs: Dict,
                         comparison_type: str = "smoothed_energy",
                         error_metric: str = None) -> Dict:
    """Perform spatial error analysis between different RIR calculation methods.

    Args:
        room_params (dict): Room parameters
        source_pos (Tuple[float, float, float]): Source position
        duration (float): Duration of RIR
        Fs (int): Sampling frequency
        methods (List[str]): List of methods to compare
        method_configs (Dict): Configuration dictionary for each method
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
    # Generate receiver positions
    receiver_positions = generate_receiver_grid(room_params['width'], room_params['depth'])

    # Initialize room and the source signal
    room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
    num_samples = int(Fs * duration)
    impulse = geometry.Source.generate_signal('dirac', num_samples)
    room.set_source(*source_pos, signal=impulse['signal'], Fs=Fs)

    # Calculate reflection coefficient
    room.wallAttenuation = [np.sqrt(1 - room_params['absorption'])] * 6

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
        rir_methods[method] = []
        config = method_configs[method]['params']
        
        for rx, ry in receiver_positions:
            room.set_microphone(rx, ry, room_params['mic z'])

            if method == 'ISM':
                # Setup PRA room
                pra_room = pra.ShoeBox(room_dim, fs=Fs,
                                     materials=pra.Material(room_params['absorption']),
                                     max_order=config.get('max_order', 12),
                                     air_absorption=False,
                                     ray_tracing=config.get('ray_tracing', False),
                                     use_rand_ism=config.get('use_rand_ism', False))
                pra_room.set_sound_speed(343)

                # Add source and receiver
                source_loc = np.array([source_pos[0], source_pos[1], source_pos[2]])
                mic_loc = np.array([rx, ry, room_params['mic z']])
                pra_room.add_source(source_loc)
                pra_room.add_microphone(mic_loc)

                # Compute RIR
                pra_room.compute_rir()
                rir = pra_room.rir[0][0]

                # Normalize
                rir = rir / np.max(np.abs(rir))

                # Handle global delay
                global_delay = pra.constants.get("frac_delay_length") // 2
                rir = rir[global_delay:]  # Shift left by removing the initial delay
                rir = np.pad(rir, (0, global_delay))  # Pad with zeros at the end to maintain length
                rir = rir[:num_samples]  # Trim to desired length

            elif method == 'SDN-Base':
                # Create a copy of room parameters with current mic position
                current_params = room_params.copy()
                current_params.update({
                    'mic x': rx,
                    'mic y': ry
                })
                rir = calculate_sdn_base_rir(current_params, duration, Fs)

            elif method.startswith('SDN-Test'):
                sdn = DelayNetwork(room, Fs=Fs, label=method, **config)
                rir = sdn.calculate_rir(duration)
                rir = rir / np.max(np.abs(rir))

            rir_methods[method].append(rir)

    # Calculate error maps for each method comparison
    mean_errors = {}  # Dictionary to store mean errors
    
    # Assuming 'ISM' is one of the methods
    if 'ISM' not in methods:
        raise ValueError("ISM must be one of the methods for comparison")
        
    # Only compare other methods with ISM
    for method in methods:
        if method != 'ISM':
            errors = []
            for rir1, rir2 in zip(rir_methods['ISM'], rir_methods[method]):
                # Process signals based on comparison type
                if comparison_type == "smoothed_energy":
                    _, sig1 = calculate_smoothed_energy(rir1, window_length=30, range=50, Fs=Fs)
                    _, sig2 = calculate_smoothed_energy(rir2, window_length=30, range=50, Fs=Fs)
                elif comparison_type == "energy":
                    sig1, _ = calculate_smoothed_energy(rir1, window_length=30, range=50, Fs=Fs)
                    sig2, _ = calculate_smoothed_energy(rir2, window_length=30, range=50, Fs=Fs)
                elif comparison_type == "edc":
                    sig1 = compute_edc(rir1, Fs, plot=False)
                    sig2 = compute_edc(rir2, Fs, plot=False)
                else:
                    raise ValueError(f"Unknown comparison type: {comparison_type}")

                # Compute error using specified metric
                error = compute_RMS(sig1, sig2, range=50, Fs=Fs, method=error_metric)
                errors.append(error)
            
            error_map = np.array(errors).reshape(grid_size, grid_size)
            comparison_key = f'ISM_vs_{method}'
            error_maps[comparison_key] = {
                'X': X,
                'Y': Y,
                'errors': error_map
            }
            
            # Calculate and store mean error
            mean_errors[comparison_key] = np.mean(errors)
    
    # Print mean errors
    print("\nMean Errors:")
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
    
    return error_maps, rir_methods, receiver_positions

def plot_error_maps(error_maps: Dict, room_params: dict, source_pos: Tuple[float, float, float], 
                   interpolated: bool = True, comparison_type: str = "smoothed_energy", error_metric: str = "rmse"):
    """Plot error maps with room layout.
    
    Args:
        error_maps (Dict): Dictionary containing error maps
        room_params (dict): Room parameters
        source_pos (Tuple[float, float, float]): Source position
        interpolated (bool): Whether to use interpolation in the plot (default: True)
        comparison_type (str): Type of comparison used
        error_metric (str): Error metric used
    """
    # Filter for only ISM comparisons
    ism_comparisons = {k: v for k, v in error_maps.items() if k.startswith('ISM_')}
    
    if not ism_comparisons:
        print("No ISM comparisons found in error maps.")
        return
        
    n_comparisons = len(ism_comparisons)
    fig, axes = plt.subplots(n_comparisons, 1, figsize=(10, 5*n_comparisons))
    if n_comparisons == 1:
        axes = [axes]
    
    # Find global min and max for consistent colormap across subplots
    all_errors = np.concatenate([data['errors'].flatten() for data in ism_comparisons.values()])
    vmin, vmax = np.min(all_errors), np.max(all_errors)
    
    for ax, (comparison, data) in zip(axes, ism_comparisons.items()):
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
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    room_sdn_original = {
        'width': 9, 'depth': 7, 'height': 4,
        'source x': 4.5, 'source y': 3.5, 'source z': 2,
        'mic z': 1.5,
        'absorption': 0.2,
    }

    # Method configurations
    method_configs = {
        'ISM': {
            'enabled': True,
            'params': {
                'max_order': 12,
                'ray_tracing': False,
                'use_rand_ism': False
            },
            'info': ''  # Empty info for ISM
        },
        'SDN-Original': {
            'enabled': True,
            'params': {},  # No additional parameters needed
            'info': 'original implementation'
        },
        'SDN-Test1': {
            'enabled': True,
            'params': {
                'specular_source_injection': True,
                'source_weighting': 3
            },
            'info': 'source weight 3'
        },
        'SDN-Test2': {
            'enabled': True,
            'params': {
                'specular_source_injection': True,
                'source_weighting': 4
            },
            'info': 'source weight 4'
        },
        # 'SDN-Test3': {
        #     'enabled': True,
        #     'params': {
        #         'specular_source_injection': True,
        #         'source_weighting': 5
        #     },
        #     'info': 'source weight 5'
        # }
    }

    # Get list of enabled methods
    enabled_methods = [method for method, config in method_configs.items() 
                      if config['enabled']]

    room_parameters = room_sdn_original
    source_position = (room_parameters['source x'], 
                      room_parameters['source y'], 
                      room_parameters['source z'])
    
    # Analysis parameters
    USE_INTERPOLATION = False  # Set to False for discrete visualization
    
    # Example of different comparison types
    comparison_configs = [
        {"type": "smoothed_energy", "metric": "median"},
        # {"type": "smoothed_energy", "metric": "sum"},
        {"type": "edc", "metric": "rmse"},
        # {"type": "edc", "metric": "sum"}
    ]
    
    # Run analysis for each configuration
    for config in comparison_configs:
        print(f"\nRunning analysis with {config['type']} comparison and {config['metric']} metric...")
        error_maps, rir_methods, receiver_positions = spatial_error_analysis(
            room_parameters,
            source_position,
            duration=0.05,
            Fs=44100,
            methods=enabled_methods,
            method_configs=method_configs,  # Pass method configurations
            comparison_type=config['type'],
            error_metric=config['metric']
        )
        
        # Plot error maps with chosen visualization method
        plot_error_maps(error_maps, room_parameters, source_position, 
                       interpolated=USE_INTERPOLATION,
                       comparison_type=config['type'],
                       error_metric=config['metric'])
        
        # Plot RIRs for first 4 receiver positions
        # plot_rirs(rir_methods, receiver_positions, selected_positions=[0, 1, 2, 3])
