import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import analysis as an
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json

from plotting_utils import (
    load_data, 
    get_display_name, 
    DISPLAY_NAME_MAP
)

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


def print_consolidated_summary_table(all_errors: Dict, all_energies: Dict, all_rt60s: Dict, reference_method: str, source_coordinates: Dict, method_configs: Dict):
    """Prints a single consolidated summary table for RMSE, Energy, and RT60."""
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
    if show_average_col:
        for key, source_data in all_errors.items():
            if source_data:
                avg_errors[key] = np.mean(list(source_data.values()))
        
        energies_by_method, rt60s_by_method = {}, {}
        for source_name in all_source_names:
            for method, value in all_energies.get(source_name, {}).items():
                if method not in energies_by_method: energies_by_method[method] = []
                energies_by_method[method].append(value)
            for method, value in all_rt60s.get(source_name, {}).items():
                if method not in rt60s_by_method: rt60s_by_method[method] = []
                rt60s_by_method[method].append(value)

        for method, values in energies_by_method.items():
            avg_energies[method] = np.mean(values)
        for method, values in rt60s_by_method.items():
            valid_rt60s = [v for v in values if v is not None and not np.isnan(v)]
            if valid_rt60s:
                avg_rt60s[method] = np.mean(valid_rt60s)

    # --- Print Table ---
    print("\n\n" + "="*120)
    print("--- CONSOLIDATED SPATIAL ANALYSIS SUMMARY ---")
    ref_display_name = get_display_name(reference_method, method_configs, DISPLAY_NAME_MAP)
    print(f"Reference Method for RMSE: {ref_display_name}")
    print("="*120)

    # --- Header ---
    header1 = f"{'Method vs. Ref':<25}"
    header2 = f"{'':<25}"
    col_width = 27  # 8 for each metric + 3 for separators
    for source_name in all_source_names:
        display_source_name = source_name.replace('aes_', '').replace('_source', '')
        coords = source_coordinates.get(source_name)
        coord_str = f"({coords[0]:.1f},{coords[1]:.1f})" if coords is not None else ""
        header1 += f" | {display_source_name + ' ' + coord_str:<{col_width}}"
        header2 += f" | {'RMSE':>8} {'Energy':>8} {'RT60':>8} "
    
    if show_average_col:
        header1 += f" | {'Average':<{col_width}}"
        header2 += f" | {'RMSE':>8} {'Energy':>8} {'RT60':>8} "
    
    print(header1)
    print(header2)
    print("-" * len(header1))

    # --- Rows ---
    all_test_keys = sorted(list(set(k.split('_vs_')[1] for k in all_comparison_keys)))
    for test_key in all_test_keys:
        display_name = get_display_name(test_key, method_configs, DISPLAY_NAME_MAP)
        row = f"{display_name:<25}"
        
        comparison_key = f"{reference_method}_vs_{test_key}"
        
        for source_name in all_source_names:
            error = all_errors.get(comparison_key, {}).get(source_name, float('nan'))
            energy = all_energies.get(source_name, {}).get(test_key, float('nan'))
            rt60 = all_rt60s.get(source_name, {}).get(test_key, float('nan'))
            row += f" | {error:>8.3f} {energy:>8.2f} {rt60:>8.3f} "

        if show_average_col:
            avg_err = avg_errors.get(comparison_key, float('nan'))
            avg_en = avg_energies.get(test_key, float('nan'))
            avg_rt = avg_rt60s.get(test_key, float('nan'))
            row += f" | {avg_err:>8.3f} {avg_en:>8.2f} {avg_rt:>8.3f} "
            
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
    available_methods = [m for m in all_edcs.keys() if m != reference_method]
    print("\n--- Method Availability Report ---")
    print(f"Methods available in data file: {available_methods}")
    
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

    # --- Print Mean Errors and Prepare Return Data ---
    print("\n--- Mean Spatial RMSE Errors ---")
    mean_errors_dict = {}
    for comparison_key, data in error_maps.items():
        mean_error = np.mean(data['errors'])
        mean_errors_dict[comparison_key] = mean_error
        ref_method_key, test_method_key = comparison_key.split('_vs_')
        ref_display_name = DISPLAY_NAME_MAP.get(ref_method_key, ref_method_key)
        test_display_name = DISPLAY_NAME_MAP.get(test_method_key, test_method_key)
        print(f"  {ref_display_name} vs. {test_display_name}: {mean_error:.4f}")
    print("----------------------------------\n")

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

    return mean_errors_dict, X, Y, error_maps


if __name__ == "__main__":
    # --- CONFIGURATION ---
    data_dir = "results/paper_data"
    output_dir = "results/paper_figures"
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
        # "aes_room_spatial_edc_data_top_middle_source.npz",
        # "aes_room_spatial_edc_data_upper_right_source.npz",
        # "aes_room_spatial_edc_data_lower_left_source.npz",
    ]

    # --- ANALYSIS PARAMETERS ---
    REFERENCE_METHOD = 'RIMPY-neg10'
    # REFERENCE_METHOD = 'RIMPY-neg'
    # REFERENCE_METHOD = 'ISM-pra-rand10'
    # REFERENCE_METHOD = 'ISM'
    # Specify which methods to plot. Leave empty or set to None to plot all.
    METHODS_TO_PLOT = ['SDN-Test1', 'SDN-Test2', 'SDN-Test3','SDN-Test5']
    # METHODS_TO_PLOT = None

    COMPARISON_TYPE = 'edc'  # 'edc' is the most common for this
    ERROR_METRIC = 'rmse'
    ERROR_DURATION_MS = 50  # Analyze the first 50ms of the EDC

    # --- CONTROL FLAGS ---
    SAVE_FIGURES = False  # Set to True to save the generated figures to disk
    SHOW_PLOTS = False    # Set to True to display interactive plot windows
    SAVE_SUMMARY_TEXT = False # Set to False to disable saving the summary .txt file

    # --- EXECUTION LOOP ---
    all_mean_errors = {}
    source_coordinates = {}
    all_mean_energies_by_source = {}
    all_mean_rt60s_by_source = {}
    for filename in files_to_process:
        print(f"\n\n--- Processing file: {filename} ---")
        data_path = os.path.join(data_dir, filename)

        try:
            sim_data = load_data(data_path)
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

        # e.g., "fig_spatial_edc_err_aes_center_source_ref_RIMPY-neg.png"
        output_filename = f"fig_spatial_edc_err_{base_name}_ref_{REFERENCE_METHOD}.png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"--- Generating spatial error plots for: {filename} ---")
        print(f"Reference method: {REFERENCE_METHOD}")
        if SAVE_FIGURES:
            print(f"Output will be saved to: {output_path}")

        mean_errors, X, Y, err_maps = calculate_and_plot_error_maps(
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

        if mean_errors:
            for comparison_key, mean_error in mean_errors.items():
                if comparison_key not in all_mean_errors:
                    all_mean_errors[comparison_key] = {}
                all_mean_errors[comparison_key][base_name] = mean_error

    # After processing all files, print the consolidated summary table
    print_consolidated_summary_table(
        all_mean_errors,
        all_mean_energies_by_source,
        all_mean_rt60s_by_source,
        REFERENCE_METHOD,
        source_coordinates,
        sim_data.get('method_configs', {})
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
                    REFERENCE_METHOD,
                    source_coordinates,
                    sim_data.get('method_configs', {})
                )
                f.write(buf.getvalue())
        print(f"--- Saved formatted summary tables to: {text_summary_path} ---")


    print("\n--- All spatial analysis complete. ---")
