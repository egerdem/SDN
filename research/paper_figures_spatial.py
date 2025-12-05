import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
from analysis import analysis as an
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import pprint

from analysis.plotting_utils import (
    load_data, 
    get_display_name, 
    DISPLAY_NAME_MAP
)
from analysis import plot_room as pp
import sys

def plot_single_error_map(ax, data, vmin, vmax, interpolated, error_metric, room_params, source_pos, comparison_key, comparison_type, mean_error):
    """Helper function to plot a single error map on a given axes object."""

    if interpolated:
        contour = ax.contourf(data['X'], data['Y'], data['errors'], 
                            levels=20, cmap='viridis',
                            vmin=vmin, vmax=vmax)
    else:
        contour = ax.pcolormesh(data['X'], data['Y'], data['errors'],
                              cmap='viridis', shading='nearest',
                              vmin=vmin, vmax=vmax)
    
    ax.scatter(data['X'].flatten(), data['Y'].flatten(), 
              color='white', alpha=0.8, s=50,
              marker='o', edgecolor='black', linewidth=0.5,
              label='Receivers')
    
    ax.plot([0, room_params['width']], [0, 0], 'k-', linewidth=2)
    ax.plot([0, room_params['width']], [room_params['depth'], room_params['depth']], 'k-', linewidth=2)
    ax.plot([0, 0], [0, room_params['depth']], 'k-', linewidth=2)
    ax.plot([room_params['width'], room_params['width']], [0, room_params['depth']], 'k-', linewidth=2)

    
    # Change source marker to a red circle with a black border
    ax.plot(source_pos[0], source_pos[1], 
            marker='o',
            markerfacecolor='red',
            markeredgecolor='black',
            markeredgewidth=1.5,
            markersize=12,
            linestyle='None',
            label='Source')
    
    # Add error values as text on each grid point for clarity.
    # The text color is chosen dynamically for contrast against the background.
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')
    
    for i in range(data['X'].shape[0]):
        for j in range(data['X'].shape[1]):
            x_pos, y_pos = data['X'][i, j], data['Y'][i, j]
            error_val = data['errors'][i, j]
            color_val = norm(error_val)
            
            # Determine text color based on background brightness
            # (Luminance formula for RGB)
            rgb = cmap(color_val)[:3]
            luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            text_color = 'white' if luminance < 0.5 else 'black'
            offset = 0.25
            ax.text(x_pos, y_pos - offset, f'{error_val:.2g}',
                    ha='center', va='center',
                    color=text_color, fontsize=14) # text size for error values

    # Move colorbar to the top of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="4%", pad=0.1)
    cbar = plt.colorbar(contour, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=14)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    # cbar.set_label(f'{error_metric.upper()} Error', labelpad=-40)
    
    # Use display names for title
    ref_method_key, test_method_key = comparison_key.split('_vs_')

    ref_display_name = DISPLAY_NAME_MAP.get(ref_method_key, ref_method_key)
    if isinstance(ref_display_name, dict):
        ref_display_name = ref_display_name.get('name', ref_method_key)
    
    test_display_name = DISPLAY_NAME_MAP.get(test_method_key, test_method_key)
    if isinstance(test_display_name, dict):
        test_display_name = test_display_name.get('name', test_method_key)
    
    # Use ax.text for precise title placement inside the figure.
    # Coordinates are in 'axes fraction' (0,0 is bottom-left, 1,1 is top-right).
    ax.text(0.5, 0.92, f'{test_display_name}',
            transform=ax.transAxes,
            ha='center', va='top',  # Horizontal alignment: center, Vertical alignment: top
            fontsize=14, # text of title
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add mean error text to the plot, with configurable location.
    ax.text(0.94, 0.07, f'Mean Err: {mean_error:.3f}',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=14, # text of mean error
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.set_xlabel('Room Width (m)', fontsize=12)
    ax.set_ylabel('Room Depth (m)', fontsize=12)
    
    # Use bbox_to_anchor for precise legend placement to avoid overflow.
    # (x, y) coordinates are in 'axes fraction'.
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.94),
              fancybox=True, shadow=False, ncol=1, fontsize=12, #size of legend text
              bbox_transform=ax.transAxes)

    # Customize grid to be denser, gray, and dashed
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_aspect('equal', adjustable='box')


def print_consolidated_summary_table(all_errors: Dict, all_energies: Dict, all_rt60s: Dict, 
                                      all_smoothed_50ms: Dict, all_smoothed_full: Dict,
                                      reference_method: str, source_coordinates: Dict, method_configs: Dict):
    """Prints a single consolidated summary table for RMSE, Energy, RT60, and Smoothed RIR errors."""
    if not all_errors:
        print("No error results to summarize.")
        return

    all_comparison_keys = sorted(all_errors.keys())
    # All three dicts should have the same source names as keys
    all_source_names = sorted(all_energies.keys())

    if not all_source_names:
        print("No source data found in results.")
        return
    
    # The "Average" column is only meaningful if there are multiple sources to average over.
    show_average_col = len(all_source_names) > 1

    # --- Calculate Averages (only if needed) ---
    avg_errors, avg_energies, avg_rt60s = {}, {}, {}
    avg_smoothed_50ms, avg_smoothed_full = {}, {}
    if show_average_col:
        for key, source_data in all_errors.items():
            if source_data:
                avg_errors[key] = np.mean(list(source_data.values()))
        
        energies_by_method, rt60s_by_method = {}, {}
        smoothed_50ms_by_comparison, smoothed_full_by_comparison = {}, {}
        for source_name in all_source_names:
            for method, value in all_energies.get(source_name, {}).items():
                if method not in energies_by_method: energies_by_method[method] = []
                energies_by_method[method].append(value)
            for method, value in all_rt60s.get(source_name, {}).items():
                if method not in rt60s_by_method: rt60s_by_method[method] = []
                rt60s_by_method[method].append(value)
            
            # Collect smoothed RIR errors
            for comparison_key, value in all_smoothed_50ms.get(source_name, {}).items():
                if comparison_key not in smoothed_50ms_by_comparison: smoothed_50ms_by_comparison[comparison_key] = []
                smoothed_50ms_by_comparison[comparison_key].append(value)
            for comparison_key, value in all_smoothed_full.get(source_name, {}).items():
                if comparison_key not in smoothed_full_by_comparison: smoothed_full_by_comparison[comparison_key] = []
                smoothed_full_by_comparison[comparison_key].append(value)

        for method, values in energies_by_method.items():
            avg_energies[method] = np.mean(values)
        for method, values in rt60s_by_method.items():
            valid_rt60s = [v for v in values if v is not None and not np.isnan(v)]
            if valid_rt60s:
                avg_rt60s[method] = np.mean(valid_rt60s)
        for comparison_key, values in smoothed_50ms_by_comparison.items():
            avg_smoothed_50ms[comparison_key] = np.mean(values)
        for comparison_key, values in smoothed_full_by_comparison.items():
            avg_smoothed_full[comparison_key] = np.mean(values)

    # --- Print Table ---
    print("\n\n" + "="*120)
    print("--- CONSOLIDATED SPATIAL ANALYSIS SUMMARY ---")
    ref_display_name = get_display_name(reference_method, method_configs, DISPLAY_NAME_MAP)
    print(f"Reference Method for RMSE: {ref_display_name}")
    print("="*120)

    # --- Header ---
    method_col_width = 40  # Wider column for method names to ensure alignment
    header1 = f"{'Method vs. Ref':<{method_col_width}}"
    header2 = f"{'':<{method_col_width}}"
    col_width = 44  # Updated for 5 metrics: EDC-RMSE, Energy, RT60, Smooth-50ms, Smooth-Full
    for source_name in all_source_names:
        display_source_name = source_name.replace('aes_', '').replace('_source', '')
        coords = source_coordinates.get(source_name)
        coord_str = f"({coords[0]:.1f},{coords[1]:.1f})" if coords is not None else ""
        header1 += f" | {display_source_name + ' ' + coord_str:<{col_width}}"
        header2 += f" | {'EDC':>7} {'Energy':>7} {'RT60':>6} {'Sm-50ms':>9} {'Sm-Full':>9} "
    
    if show_average_col:
        header1 += f" | {'Average':<{col_width}}"
        header2 += f" | {'EDC':>7} {'Energy':>7} {'RT60':>6} {'Sm-50ms':>9} {'Sm-Full':>9} "
    
    print(header1)
    print(header2)
    print("-" * len(header1))

    # --- Rows ---
    all_test_keys = sorted(list(set(k.split('_vs_')[1] for k in all_comparison_keys)))
    for test_key in all_test_keys:
        display_name = get_display_name(test_key, method_configs, DISPLAY_NAME_MAP)
        # Truncate if too long, but ensure consistent width
        if len(display_name) > method_col_width - 1:
            display_name = display_name[:method_col_width - 4] + "..."
        row = f"{display_name:<{method_col_width}}"
        
        comparison_key = f"{reference_method}_vs_{test_key}"
        
        for source_name in all_source_names:
            error = all_errors.get(comparison_key, {}).get(source_name, float('nan'))
            energy = all_energies.get(source_name, {}).get(test_key, float('nan'))
            rt60 = all_rt60s.get(source_name, {}).get(test_key, float('nan'))
            smooth_50ms = all_smoothed_50ms.get(source_name, {}).get(comparison_key, float('nan'))
            smooth_full = all_smoothed_full.get(source_name, {}).get(comparison_key, float('nan'))
            row += f" | {error:>7.3f} {energy:>7.2f} {rt60:>6.2f} {smooth_50ms:>9.4f} {smooth_full:>9.4f} "

        if show_average_col:
            avg_err = avg_errors.get(comparison_key, float('nan'))
            avg_en = avg_energies.get(test_key, float('nan'))
            avg_rt = avg_rt60s.get(test_key, float('nan'))
            avg_s50 = avg_smoothed_50ms.get(comparison_key, float('nan'))
            avg_sf = avg_smoothed_full.get(comparison_key, float('nan'))
            row += f" | {avg_err:>7.3f} {avg_en:>7.2f} {avg_rt:>6.2f} {avg_s50:>9.4f} {avg_sf:>9.4f} "
            
        print(row)

    print("-" * len(header1))
    print("="*120)


def calculate_and_plot_error_maps(sim_data, output_path: str, reference_method: str,
                                    methods_to_plot: List[str] = None,
                                    comparison_type: str = "edc", error_metric: str = "rmse",
                                    err_duration_ms: float = 50.0, interpolated: bool = True,
                                    save_figure: bool = True, show_plot: bool = True):
    """
    Loads pre-calculated spatial data, computes error maps against a reference
    method, and plots the results. Can filter for specific methods.
    """

    # Extract data from the loaded dictionary
    receiver_positions = sim_data['receiver_positions']
    room_params = sim_data['room_params']
 
    source_pos = sim_data['source_pos']
    Fs = sim_data['Fs']
    all_edcs = sim_data.get('edcs', {})
    method_configs = sim_data.get('method_configs', {})

    if not all_edcs:
        print("Error: No EDC data found in the file.")
        return
        
    if reference_method not in all_edcs:
        print(f"Error: Reference method '{reference_method}' not found in the data file.")
        print(f"Available methods: {list(all_edcs.keys())}")
        return

    # --- Error Calculation ---
    print(f"\n--- Calculating {error_metric.upper()} for {comparison_type} (first {err_duration_ms}ms) ---")
    
    error_maps = {}
    grid_size = int(np.sqrt(len(receiver_positions)))
    X = receiver_positions[:, 0].reshape(grid_size, grid_size)
    Y = receiver_positions[:, 1].reshape(grid_size, grid_size)
    
    ref_signals = all_edcs[reference_method]

    # --- Method Filtering and Availability Report ---
    available_methods = [m for m in all_edcs.keys()]
    print("\n--- Method Availability Report ---")
    pprint.pp(f"Methods available in data file: {available_methods}")
    
    methods_to_compare = available_methods
    
    # If a specific list of methods is provided, filter for them
    if methods_to_plot:
        print(f"Methods requested for plotting: {methods_to_plot}")
        # Intersect available methods with requested methods
        final_methods_to_plot = [m for m in methods_to_plot if m in available_methods]
        
        # Report on what will actually be plotted
        print(f"==> Final methods to be plotted: {final_methods_to_plot}")
        
        # Report on any requested methods that were not found
        skipped_methods = [m for m in methods_to_plot if m not in available_methods]
        if skipped_methods:
            print(f"==> Skipping requested methods not found in file: {skipped_methods}")

        methods_to_compare = final_methods_to_plot
    else:
        print(f"==> Plotting all available methods: {methods_to_compare}")
    print("------------------------------------")

    for method in methods_to_compare:
        print(f"  Comparing '{reference_method}' vs '{method}'...")
        comparison_key = f'{reference_method}_vs_{method}'
        error_maps[comparison_key] = {'X': X, 'Y': Y, 'errors': []}

        test_signals = all_edcs[method]

        for i in range(len(receiver_positions)):
            sig1 = ref_signals[i]
            sig2 = test_signals[i]
            
            # Using the same robust RMSE calculation from spatial_analysis
            error = an.compute_RMS(
                sig1, 
                sig2, 
                range=int(err_duration_ms),
                Fs=Fs,
                skip_initial_zeros=True,
                normalize_by_active_length=True
            )
            error_maps[comparison_key]['errors'].append(error)
        
        error_maps[comparison_key]['errors'] = np.array(error_maps[comparison_key]['errors']).reshape(grid_size, grid_size)

    if not error_maps:
        print("No methods to compare against the reference. Nothing to plot.")
        return None

    # --- Print Mean Errors and Per-Receiver Errors ---
    print("\n--- Mean Spatial RMSE Errors ---")
    mean_errors_dict = {}
    per_receiver_errors_dict = {}  # Store per-receiver errors
    for comparison_key, data in error_maps.items():
        mean_error = np.mean(data['errors'])
        mean_errors_dict[comparison_key] = mean_error
        per_receiver_errors_dict[comparison_key] = data['errors'].flatten()  # Store individual errors
        ref_method_key, test_method_key = comparison_key.split('_vs_')
        ref_display_name = DISPLAY_NAME_MAP.get(ref_method_key, ref_method_key)
        test_display_name = DISPLAY_NAME_MAP.get(test_method_key, test_method_key)
        print(f"  {ref_display_name} vs. {test_display_name}: {mean_error:.4f}")
    print("----------------------------------\n")
    
    # --- Print Per-Receiver RMSE Table ---
    print("\n--- Per-Receiver RMSE Errors ---")
    grid_size = int(np.sqrt(len(receiver_positions)))
    # Print header
    header = f"{'Receiver':<15} {'Position (x,y)':<20}"
    for comparison_key in sorted(error_maps.keys()):
        ref_method_key, test_method_key = comparison_key.split('_vs_')
        test_display_name = DISPLAY_NAME_MAP.get(test_method_key, test_method_key)
        if isinstance(test_display_name, dict):
            test_display_name = test_display_name.get('name', test_method_key)
        # Truncate long names
        if len(test_display_name) > 20:
            test_display_name = test_display_name[:17] + "..."
        header += f" | {test_display_name:>20}"
    print(header)
    print("-" * len(header))
    
    # Print rows for each receiver
    for i in range(len(receiver_positions)):
        rx, ry = receiver_positions[i]
        row = f"Receiver {i+1:2d}     ({rx:.2f},{ry:.2f})"
        for comparison_key in sorted(error_maps.keys()):
            error_val = per_receiver_errors_dict[comparison_key][i]
            row += f" | {error_val:>20.6f}"
        print(row)
    
    # Print mean row
    mean_row = f"{'Mean RMSE':<15} {'':<20}"
    for comparison_key in sorted(error_maps.keys()):
        mean_error = mean_errors_dict[comparison_key]
        mean_row += f" | {mean_error:>20.6f}"
    print("-" * len(header))
    print(mean_row)
    print("=" * len(header) + "\n")

    # --- Plotting ---
    n_comparisons = len(error_maps)
    # Arrange plots horizontally: 1 row, n_comparisons columns
    fig, axes = plt.subplots(1, n_comparisons, figsize=(7 * n_comparisons, 8), squeeze=False)
    axes = axes.flatten()

    all_errors = np.concatenate([data['errors'].flatten() for data in error_maps.values()])
    vmin, vmax = np.min(all_errors), np.max(all_errors)
    
    for i, (comparison_key, data) in enumerate(error_maps.items()):
        mean_error = np.mean(data['errors'])
        plot_single_error_map(axes[i], data, vmin, vmax, interpolated, error_metric,
                                room_params, source_pos, comparison_key, comparison_type, mean_error)

    fig.tight_layout()
    if save_figure:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n--- Saved spatial error map to: {output_path} ---")

    if show_plot:
        plt.show()
    else:
        plt.close(fig) # Avoid displaying and free up memory

    return mean_errors_dict, X, Y, error_maps, per_receiver_errors_dict


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Use absolute paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    data_dir = os.path.join(project_root, "results", "paper_data")
    output_dir = os.path.join(project_root, "results", "paper_figures")
    os.makedirs(output_dir, exist_ok=True)

    # --- SELECT DATA FILES ---
    # List of data files to process. This will generate a separate figure for each file.
    files_to_process = [
        # "aes_room_spatial_edc_data_upper_right_source.npz",
        # "aes_room_spatial_edc_data_upper_right_sourcev2_8_6_2.npz",
        # "aes_room_spatial_edc_data_upper_right_sourcev2_7d5_6_2.npz"
    ]

    # files_to_process = ["journal_room_spatial_edc_data.npz"]  # Single file for now, can be expanded later

    files_to_process = [
        "aes_room_spatial_edc_data_center_source.npz",
        "aes_room_spatial_edc_data_top_middle_source.npz",
        "aes_room_spatial_edc_data_upper_right_source.npz",
        "aes_room_spatial_edc_data_lower_left_source.npz",
    ]

    # --- ANALYSIS PARAMETERS ---
    REFERENCE_METHOD = 'RIMPY-neg10'
    reference_method = REFERENCE_METHOD
    # REFERENCE_METHOD = 'RIMPY-neg'
    # REFERENCE_METHOD = 'ISM-pra-rand10'
    # REFERENCE_METHOD = 'ISM'
    # Specify which methods to plot. Leave empty or set to None to plot all.
    METHODS_TO_PLOT = ['SDN-Test_3', 'SDN-Test_2', 'SDN-Test1', 'SDN-Test2', 'SDN-Test3', 'SDN-Test4','SDN-Test5',
                       'SDN-Test6', 'SDN-Test7']
    # METHODS_TO_PLOT = ["SDN-c_center", "SDN-c_lower_left", 'SDN-Test2.998', "SDN-c_upper_right"]
    # METHODS_TO_PLOT = ['SDN-Test2.998']
    # METHODS_TO_PLOT = ['SDN-fast4_71'] #not fast actually, fyi. name wrong.
    # METHODS_TO_PLOT = None

    COMPARISON_TYPE = 'edc'  # 'edc' is the most common for this
    ERROR_METRIC = 'rmse'
    ERROR_DURATION_MS = 50  # Analyze the first 50ms of the EDC

    # --- CONTROL FLAGS ---
    SAVE_FIGURES = False  # Set to True to save the generated figures to disk
    SHOW_PLOTS = False    # Set to True to display interactive plot windows
    SAVE_SUMMARY_TEXT = False # Set to False to disable saving the summary .txt file
    SHOW_INTERACTIVE_PLOT = True  # Set to True to show unified interactive RIR plot for first receiver

    # --- EXECUTION LOOP ---
    all_mean_errors = {}
    source_coordinates = {}
    all_mean_energies_by_source = {}
    all_mean_rt60s_by_source = {}
    all_smoothed_50ms_by_source = {}
    all_smoothed_full_by_source = {}
    for filename in files_to_process:
        print(f"\n\n--- Processing file: {filename} ---")
        data_path = os.path.join(data_dir, filename)
        try:
            sim_data = load_data(data_path)
            
            # --- Print Experiment Info ---
            print("\n--- Experiment Information ---")
            room_params = sim_data.get('room_params', {})
            print(f"Room: {room_params.get('display_name', 'Unknown')}")
            print(f"Dimensions: {room_params.get('width')}x{room_params.get('depth')}x{room_params.get('height')}")
            
            print(f"Source Position: {sim_data.get('source_pos')}")
            
            receivers = sim_data.get('receiver_positions', [])
            print(f"Number of Receivers: {len(receivers)}")
            print("receiver positions (x, y, z):", receivers)

        except FileNotFoundError as e:
            print(f"Error: {e}. Skipping this file.")
            continue

        # Generate the new, structured output filename
        # e.g., "aes_center_source"
        base_name = filename.replace('.npz', '').replace('_spatial_edc_data', '').replace('_room', '')
        source_coordinates[base_name] = sim_data['source_pos']
        
        # --- Calculate Mean RIR Energy for the current source ---
        mean_energies_for_current_source = {}
        if 'rirs' in sim_data:
            for key, rirs_data in sim_data["rirs"].items():
                energies = []
                if rirs_data.ndim == 2:  # Handles a grid of receivers
                    for rir_1d in rirs_data:
                        _, energy, _ = an.calculate_err(rir_1d)
                        energies.append(np.sum(energy))
                elif rirs_data.ndim == 1:  # Handles a single RIR
                    _, energy, _ = an.calculate_err(rirs_data)
                    energies.append(np.sum(energy))
                
                if energies:
                    mean_energies_for_current_source[key] = np.mean(energies)

        all_mean_energies_by_source[base_name] = mean_energies_for_current_source
        
        # --- Calculate Mean RT60 for the current source ---
        rt60s_for_current_source = {}
        if 'rirs' in sim_data:
            for key, rirs_data in sim_data["rirs"].items():
                rt60_values = []
                # Ensure we handle both single (1D) and multiple (2D) RIRs
                if rirs_data.ndim == 2:
                    for rir_1d in rirs_data:
                        if np.any(rir_1d): # Check if rir is not all zeros
                            rt60 = an.calculate_rt60_from_rir(rir_1d, sim_data['Fs'], plot=False)
                            rt60_values.append(rt60)
                elif rirs_data.ndim == 1:
                    if np.any(rirs_data):
                        rt60 = an.calculate_rt60_from_rir(rirs_data, sim_data['Fs'], plot=False)
                        rt60_values.append(rt60)

                if rt60_values:
                    # Filter out potential None values before calculating mean
                    valid_rt60s = [rt for rt in rt60_values if rt is not None]
                    if valid_rt60s:
                        rt60s_for_current_source[key] = np.mean(valid_rt60s)

        all_mean_rt60s_by_source[base_name] = rt60s_for_current_source

        # --- Calculate Smoothed RIR Errors (50ms and full) for the current source ---
        smoothed_50ms_errors = {}
        smoothed_full_errors = {}
        if 'rirs' in sim_data and reference_method in sim_data['rirs']:
            ref_rirs = sim_data['rirs'][reference_method]
            Fs = sim_data['Fs']
            
            for method_key, test_rirs in sim_data['rirs'].items():
                if method_key == reference_method:
                    continue
                
                comparison_key = f"{reference_method}_vs_{method_key}"
                errors_50ms = []
                errors_full = []
                
                # Handle both 2D (grid) and 1D (single) RIR arrays
                if ref_rirs.ndim == 2 and test_rirs.ndim == 2:
                    for i in range(len(ref_rirs)):
                        # Calculate smoothed energy for both RIRs
                        smooth_ref = an.calculate_smoothed_energy(ref_rirs[i], window_length=30, Fs=Fs)
                        smooth_test = an.calculate_smoothed_energy(test_rirs[i], window_length=30, Fs=Fs)
                        
                        # 50ms error
                        err_samples_50ms = int(50 / 1000 * Fs)
                        error_50ms = an.compute_RMS(
                            smooth_ref[:err_samples_50ms], 
                            smooth_test[:err_samples_50ms],
                            range=50, Fs=Fs,
                            skip_initial_zeros=True,
                            normalize_by_active_length=True
                        )
                        errors_50ms.append(error_50ms)
                        
                        # Full RIR error
                        error_full = an.compute_RMS(
                            smooth_ref, smooth_test,
                            range=None, Fs=Fs,
                            skip_initial_zeros=True,
                            normalize_by_active_length=True
                        )
                        errors_full.append(error_full)
                
                if errors_50ms:
                    smoothed_50ms_errors[comparison_key] = np.mean(errors_50ms)
                if errors_full:
                    smoothed_full_errors[comparison_key] = np.mean(errors_full)
        
        all_smoothed_50ms_by_source[base_name] = smoothed_50ms_errors
        all_smoothed_full_by_source[base_name] = smoothed_full_errors

        # e.g., "fig_spatial_edc_err_aes_center_source_ref_RIMPY-neg.png"
        output_filename = f"fig_spatial_edc_err_{base_name}_ref_{REFERENCE_METHOD}.png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"--- Generating spatial error plots for: {filename} ---")
        print(f"Reference method: {REFERENCE_METHOD}")
        if SAVE_FIGURES:
            print(f"Output will be saved to: {output_path}")

        result = calculate_and_plot_error_maps(
            sim_data=sim_data,
            output_path=output_path,
            reference_method=REFERENCE_METHOD,
            methods_to_plot= METHODS_TO_PLOT,
            comparison_type=COMPARISON_TYPE,
            error_metric=ERROR_METRIC,
            err_duration_ms=ERROR_DURATION_MS,
            interpolated=False,
            save_figure=SAVE_FIGURES,
            show_plot=SHOW_PLOTS
        )

        if result is not None:
            mean_errors, X, Y, err_maps, per_receiver_errors = result
        else:
            mean_errors = None

        if mean_errors:
            for comparison_key, mean_error in mean_errors.items():
                if comparison_key not in all_mean_errors:
                    all_mean_errors[comparison_key] = {}
                all_mean_errors[comparison_key][base_name] = mean_error
        
        # --- Interactive RIR Plot for First Receiver ---
        if SHOW_INTERACTIVE_PLOT and 'rirs' in sim_data:
            print(f"\n--- Creating interactive RIR plot for first receiver ---")
            
            # Extract RIRs for the first receiver (index 0)
            # Only include reference method and METHODS_TO_PLOT
            first_receiver_rirs = {}
            for method_key, rirs_data in sim_data['rirs'].items():
                # Filter: only show reference method and methods in METHODS_TO_PLOT
                if method_key == REFERENCE_METHOD or (METHODS_TO_PLOT and method_key in METHODS_TO_PLOT):
                    if rirs_data.ndim == 2:  # Multi-receiver grid
                        first_receiver_rirs[method_key] = rirs_data[0]  # Get first receiver
                    elif rirs_data.ndim == 1:  # Single receiver
                        first_receiver_rirs[method_key] = rirs_data
            
            if first_receiver_rirs:
                # Reverse the order to match main.py pattern (last added shown first)
                reversed_rirs = dict(reversed(list(first_receiver_rirs.items())))
                
                # Get first receiver coordinates
                rx, ry = sim_data['receiver_positions'][0]
                print(f"First receiver position: ({rx:.2f}m, {ry:.2f}m)")
                
                # Update room parameters with first receiver position
                plot_room_params = sim_data['room_params'].copy()
                plot_room_params['mic x'] = rx
                plot_room_params['mic y'] = ry
                
                # Create the unified interactive plot
                pp.create_unified_interactive_plot(
                    reversed_rirs, 
                    sim_data['Fs'], 
                    plot_room_params,
                    reflection_times=None  # Can add reflection times if available
                )
                plt.show(block=False)
                print("Interactive plot displayed. Close the plot window to continue.")

    # After processing all files, print the consolidated summary table
    print_consolidated_summary_table(
        all_mean_errors,
        all_mean_energies_by_source,
        all_mean_rt60s_by_source,
        all_smoothed_50ms_by_source,
        all_smoothed_full_by_source,
        REFERENCE_METHOD,
        source_coordinates,
        sim_data.get('method_configs')
    )

    # --- Export Results to Files if plotting all methods ---
    if METHODS_TO_PLOT is None and SAVE_SUMMARY_TEXT:
        # Export Formatted Tables to a single .txt file
        import io
        from contextlib import redirect_stdout

        # Make filename descriptive by including room name and experiment type
        room_name = sim_data['room_params'].get('display_name', 'unknown_room').lower().replace(' ', '_')
        num_receivers = len(sim_data['receiver_positions'])
        exp_type = 'single' if num_receivers == 1 else 'multi'
        text_summary_filename = f'summary_tables_{room_name}_{exp_type}_ref_{REFERENCE_METHOD}.txt'
        text_summary_path = os.path.join(output_dir, text_summary_filename)
        
        with open(text_summary_path, 'w') as f:
            # Add a header to the file
            f.write(f"--- SUMMARY TABLES ---\n")
            f.write(f"Room: {sim_data['room_params'].get('display_name', 'N/A')}\n")
            
            # Add source position
            source_pos = sim_data['source_pos']
            f.write(f"Source Position (x,y,z): ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f})\n")

            # Add receiver position(s)
            f.write(f"Experiment Type: {exp_type}-receiver ({num_receivers} position(s))\n")
            mic_z = sim_data['room_params'].get('mic z', 'N/A')
            if exp_type == 'single':
                rec_pos = sim_data['receiver_positions'][0]
                f.write(f"Receiver Position (x,y,z): ({rec_pos[0]:.2f}, {rec_pos[1]:.2f}, {mic_z})\n")
            else:
                f.write(f"Receiver Positions: Grid of {num_receivers} positions at height z={mic_z}\n")

            f.write(f"Reference Method: {REFERENCE_METHOD}\n")
            f.write(f"----------------------\n\n")

            # Capture the output of the summary functions
            with io.StringIO() as buf, redirect_stdout(buf):
                print_consolidated_summary_table(
                    all_mean_errors,
                    all_mean_energies_by_source,
                    all_mean_rt60s_by_source,
                    all_smoothed_50ms_by_source,
                    all_smoothed_full_by_source,
                    REFERENCE_METHOD,
                    source_coordinates,
                    sim_data.get('method_configs', {})
                )
                f.write(buf.getvalue())
        print(f"--- Saved formatted summary tables to: {text_summary_path} ---")


    print("\n--- All spatial analysis complete. ---")
