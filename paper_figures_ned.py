import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List

# Import shared functions and variables from our new utility module
from plotting_utils import (
    load_data, 
    get_display_name, 
    get_linestyle, 
    PLOT_CONFIG,
    ROOM_AES,
    ROOM_JOURNAL,
    ROOM_WASPAA
)

def _plot_reflection_vertical_lines(ax, reflection_times):
    """Helper to draw vertical lines for reflection arrival times."""
    if not reflection_times:
        return

    colors = ['#d62728', '#2ca02c', '#1f77b4']  # More distinct colors
    order_map = {
        'first_order': 1,
        'second_order': 2,
        'third_order': 3
    }
    
    for order_name, time in reflection_times.items():
        if time is not None:
            order_num = order_map.get(order_name)
            if order_num is not None and order_num <= len(colors):
                color = colors[order_num - 1]
                
                # Add vertical line with a label for the legend
                ax.axvline(x=time, color=color, linestyle='--', alpha=0.6, 
                           label=f'ISM N={order_num}' if ax.get_label() != '_nolegend_' else None)
                
                # Add text annotation just above the x-axis
                ax.text(time, ax.get_ylim()[0], f'N={order_num}',
                        color=color, alpha=0.8,
                        ha='center', va='bottom', backgroundcolor='white',
                        fontsize='small', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1))

if __name__ == "__main__":
    # --- Configuration ---
    # Choose which room's data to plot by setting the active_room.
    # active_room = ROOM_AES
    active_room = ROOM_WASPAA
    # active_room = ROOM_JOURNAL

    # import path_tracker
    # room = simulation_data['room_params']
    # path_tracker = path_tracker.PathTracker()
    # ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
    # ism_calc.set_path_tracker(path_tracker)
    # ism_calc.analyze_paths(max_order=3, print_invalid=False)
    #
    # # Get arrival times for each order
    # arrival_times = path_tracker.get_latest_arrival_time_by_order('ISM')
    # reflection_times = {
    #     'first_order': arrival_times.get(1),
    #     'second_order': arrival_times.get(2),
    #     'third_order': arrival_times.get(3)
    # }

    # --- Standardized Filename Generation ---
    room_name = active_room.get('display_name', 'unknown_room')
    filename_suffix = room_name.lower().replace(' ', '_')
    data_filename = f"results/paper_data/{filename_suffix}_spatial_edc_data.npz"

    # Define the display names and the desired plot order
    display_name_and_order = {
        'ISM': 'ISM (PRA)',
        'RIMPY-neg': 'ISM (randomized )',
        'SDN-Test1': 'SDN Original (c=1)',
        'SDN-Test_3': 'SW-SDN (c=-3)',
        'SDN-Test_2': 'SW-SDN (c=-2)',
        'SDN-Test2': 'SW-SDN (c=2)',
        'SDN-Test3': 'SW-SDN (c=3)',
        'SDN-Test4': 'SW-SDN (c=4)',
        'SDN-Test5': 'SW-SDN (c=5)',
        'SDN-Test6': 'SW-SDN (c=6)',
        'SDN-Test7': 'SW-SDN (c=7)',
        'HO-SDN-N2': 'HO-SDN (N=2)',
        'HO-SDN-N3': 'HO-SDN (N=3)',
    }
    
    # excluded_methods = ["ISM"]
    excluded_methods = []

    # --- Data Loading ---
    try:
        simulation_data = load_data(data_filename)
    except FileNotFoundError as e:
        print(e)
        print("no file:", simulation_data)
        exit()

    Fs = simulation_data['Fs']
    method_configs = simulation_data['method_configs']
    
    # Extract reflection times, making sure it's a dictionary
    reflection_times_data = simulation_data.get('reflection_times')
    if reflection_times_data is not None:
        reflection_times = reflection_times_data.item()
    else:
        reflection_times = None

    # --- Data Verification and Manifest ---
    print("\n--- Plotting Manifest ---")
    print(f"{'Programmatic Key':<20} | {'Config Info':<25} | {'Plot Display Name':<30}")
    print("-" * 80)

    plot_data = {}
    # Use the order from the name map to process data
    for method_key in display_name_and_order.keys():
        if method_key in simulation_data['neds']:
            if method_key not in excluded_methods:
                # We assume single mic position, so we take the first NED profile
                avg_ned = np.mean(simulation_data['neds'][method_key], axis=0)
                plot_data[method_key] = avg_ned
                
                config_info = method_configs.get(method_key, {}).get('info', 'N/A')
                display_name = get_display_name(method_key, method_configs, display_name_and_order)
                print(f"{method_key:<20} | {config_info:<25} | {display_name:<30}")

    print("-" * 80, "\n")
    if not plot_data:
        print("Error: No NED data found or all methods are excluded/missing.")
        exit()

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(7, 6))

    for method in display_name_and_order.keys():
        if method in plot_data:
            ned_profile = plot_data[method]
            display_name = get_display_name(method, method_configs, display_name_and_order)
            linestyle = get_linestyle(method)
            time_axis = np.arange(len(ned_profile)) / Fs
            ax.plot(time_axis, ned_profile, label=display_name, linestyle=linestyle, linewidth=2)

    # --- Main Plot Formatting ---
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Normalized Echo Density', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Get plot configurations from the central utility file
    room_plot_config = PLOT_CONFIG.get(room_name, {}).get('ned', {})
    main_xlim = room_plot_config.get('main_xlim')
    main_ylim = room_plot_config.get('main_ylim')

    if main_xlim:
        ax.set_xlim(main_xlim)
    if main_ylim:
        ax.set_ylim(main_ylim)

    # Add reflection order lines to the main plot
    if reflection_times:
        _plot_reflection_vertical_lines(ax, reflection_times)

    # --- Inset Plot (using parameters from plotting_utils) ---
    if room_plot_config:
        # Check if inset parameters are defined to avoid errors
        inset_rect = room_plot_config.get('inset_rect')
        if inset_rect:
            axins = ax.inset_axes(inset_rect)
            for method in display_name_and_order.keys():
                if method in plot_data:
                    ned_profile = plot_data[method]
                    linestyle = get_linestyle(method)
                    time_axis = np.arange(len(ned_profile)) / Fs
                    axins.plot(time_axis, ned_profile, linestyle=linestyle, linewidth=1.5)
            
            # Add reflection order lines to the inset plot
            if reflection_times:
                _plot_reflection_vertical_lines(axins, reflection_times)

            axins.set_xlim(room_plot_config.get('inset_xlim'))
            axins.set_ylim(room_plot_config.get('inset_ylim'))
            axins.grid(True, linestyle='--', alpha=0.6)
            ax.indicate_inset_zoom(axins, edgecolor="black")
    else:
        print(f"Warning: No plot configuration found for '{room_name}' in plotting_utils.py")

    # --- Legend and Layout ---
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=4, fontsize='small')
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    # --- Save and Show Figure ---
    output_dir = "results/paper_figures"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"fig_ned_comparison_{filename_suffix}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved NED plot to: {output_path}")
    plt.show() 