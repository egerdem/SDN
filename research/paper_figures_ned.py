import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List

# Import shared functions and variables from our new utility module
from analysis.plotting_utils import (
    load_data, 
    get_display_name, 
    get_linestyle, 
    PLOT_CONFIG,
    DISPLAY_NAME_MAP,
    ROOM_AES,
    ROOM_JOURNAL,
    ROOM_WASPAA,
    plot_reflection_vertical_lines
)

if __name__ == "__main__":
    # --- FIGURE GENERATION SETUP ---
    data_dir = "../results/paper_data"
    output_dir = "../results/paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Load the pre-calculated reflection times for all rooms
    reflection_times_path = os.path.join(data_dir, "reflection_times.npz")
    try:
        all_reflection_times = np.load(reflection_times_path, allow_pickle=True)['all_room_data'].item()
    except FileNotFoundError:
        print("Warning: reflection_times.npz not found. Reflection time lines will not be plotted.")
        all_reflection_times = {}

    # Discover all data files to process, excluding the reflection times file
    files_to_run = [f for f in os.listdir(data_dir) if f.endswith('.npz') and f != 'reflection_times.npz']

    print(f"--- Starting NED figure generation for {len(files_to_run)} data file(s) ---")

    # --- Loop through each data file ---
    for filename in files_to_run:
        print(f"\n>>> Processing file: {filename}")
        data_path = os.path.join(data_dir, filename)

        # --- Data Loading ---
        try:
            simulation_data = load_data(data_path)
        except FileNotFoundError as e:
            print(f"  Error: {e}. Skipping.")
            continue

        room_params = simulation_data['room_params']
        if isinstance(room_params, np.ndarray):
            room_params = room_params.item()
        room_name = room_params.get('display_name', 'unknown_room')

        # Define the display names and the desired plot order
        display_name_and_order = DISPLAY_NAME_MAP
        
        excluded_methods = []

        Fs = simulation_data['Fs']
        method_configs = simulation_data['method_configs']
        
        # Get reflection times for the current room
        reflection_times = all_reflection_times.get(room_name, {}).get('reflection_times')

        # --- Data Verification and Manifest ---
        print(f"\n--- Plotting Manifest for {room_name} ---")
        print(f"{'Programmatic Key':<20} | {'Config Info':<25} | {'Plot Display Name':<30}")
        print("-" * 80)

        plot_data = {}
        for method_key in display_name_and_order.keys():
            if method_key in simulation_data['neds']:
                if method_key not in excluded_methods:
                    avg_ned = np.mean(simulation_data['neds'][method_key], axis=0)
                    plot_data[method_key] = avg_ned
                    
                    config_info = method_configs.get(method_key, {}).get('info', 'N/A')
                    display_name = get_display_name(method_key, method_configs, display_name_and_order)
                    print(f"{method_key:<20} | {config_info:<25} | {display_name:<30}")

        print("-" * 80)
        if not plot_data:
            print(f"Warning: No NED data found for {filename}. Skipping plot generation.")
            continue

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-paper')
        fig, ax = plt.subplots(figsize=(7, 6))

        for method in display_name_and_order.keys():
            if method in plot_data:
                ned_profile = plot_data[method]
                display_name = get_display_name(method, method_configs, display_name_and_order)
                linestyle = get_linestyle(method)
                time_axis = np.arange(len(ned_profile)) / Fs
                ax.plot(time_axis, ned_profile, label=display_name, linestyle=linestyle, linewidth=1.5) # main


        # --- Main Plot Formatting ---
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Normalized Echo Density', fontsize=10)
        ax.grid(True, which='both', linestyle='--',  alpha=0.6, linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        room_plot_config = PLOT_CONFIG.get(room_name, {}).get('ned', {})
        if room_plot_config.get('main_xlim'): ax.set_xlim(room_plot_config['main_xlim'])
        if room_plot_config.get('main_ylim'): ax.set_ylim(room_plot_config['main_ylim'])

        if reflection_times:
            plot_reflection_vertical_lines(ax, reflection_times)

        # --- Inset Plot ---
        if room_plot_config and room_plot_config.get('inset_rect'):
            axins = ax.inset_axes(room_plot_config['inset_rect'])
            for method in display_name_and_order.keys():
                if method in plot_data:
                    ned_profile = plot_data[method]
                    linestyle = get_linestyle(method)
                    time_axis = np.arange(len(ned_profile)) / Fs
                    axins.plot(time_axis, ned_profile, linestyle=linestyle, linewidth=1.5) # inset
            
            if room_plot_config.get('inset_xlim'): axins.set_xlim(room_plot_config['inset_xlim'])
            if room_plot_config.get('inset_ylim'): axins.set_ylim(room_plot_config['inset_ylim'])
            axins.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
            ax.indicate_inset_zoom(axins, edgecolor="black")

        # --- Legend and Layout ---
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=4, fontsize=8.5)
        fig.tight_layout(rect=[0, 0.05, 1, 1])

        # --- Save and Show Figure ---
        filename_suffix = room_name.lower().replace(' ', '_')
        output_filename = f"fig_ned_comparison_{filename_suffix}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved NED plot to: {output_path}")
        plt.show()

    print("\n--- All NED figures generated. ---") 