import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

# Import shared functions and variables from our new utility module
from analysis.plotting_utils import (
    load_data, 
    get_display_name, 
    get_linestyle, 
    get_color,
    PLOT_CONFIG,
    DISPLAY_NAME_MAP,
    plot_reflection_vertical_lines
)

def plot_edc_with_inset(data: dict, receiver_index: int, output_path: str, 
                        excluded_methods: list = None, focus_on_early: bool = True,
                        name_map: dict = None, plot_config: dict = None,
                        reflection_times: dict = None):
    """
    Generate a publication-quality EDC plot with an inset zoom, using data from
    pre-calculated data for a specific receiver position.
    Can switch between two modes: focusing on early reflections or showing the full view.
    """
    if excluded_methods is None: excluded_methods = []
    if name_map is None: name_map = {}
    if plot_config is None: plot_config = {}

    Fs = data['Fs']
    duration = data['duration']
    method_configs = data['method_configs']
    
    plt.style.use('seaborn-v0_8-paper')
    # Use the same figure size as paper_figures_ned.py for consistency
    fig, ax = plt.subplots(figsize=(7, 6))
    
    print(f"Plotting EDCs for receiver position {receiver_index}...")
    

    plotted_methods = []
    for method_key in name_map.keys():
        if method_key in data['edcs'] and method_key not in excluded_methods:
            plotted_methods.append(method_key)
            config_info = method_configs.get(method_key, {}).get('info', 'N/A')
            display_name = get_display_name(method_key, method_configs, name_map)

    
    # --- PLOTTING LOOPS ---
    for method in plotted_methods:
        edcs = data['edcs'][method]
        if receiver_index < len(edcs):
            edc = edcs[receiver_index]
            time_axis = np.arange(len(edc)) / Fs
            label = get_display_name(method, method_configs, name_map)
            linestyle = get_linestyle(method)
            color = get_color(method, name_map)
            ax.plot(time_axis, edc, label=label, linestyle=linestyle, color=color)

    # Add reflection order lines to the main plot
    if reflection_times:
        plot_reflection_vertical_lines(ax, reflection_times)

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Energy (dB)", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Use parameters from the plot_config dictionary
    inset_rect = plot_config.get('inset_rect')
    inset_xlim = plot_config.get('inset_xlim')
    inset_ylim = plot_config.get('inset_ylim')

    if focus_on_early:
        # --- MODE 1: Main plot is zoomed on early reflections ---
        main_xlim = plot_config.get('main_xlim_focused')
        main_ylim = plot_config.get('main_ylim_focused')
        ax.set_xlim(main_xlim)
        ax.set_ylim(main_ylim)
        
        # Inset shows the full duration, using the 'full' parameters from config
        axins_xlim = plot_config.get('main_xlim_full')
        axins_ylim = plot_config.get('main_ylim_full')
    else:
        # --- MODE 2: Main plot is full duration, inset is zoomed in ---
        main_xlim_full = plot_config.get('main_xlim_full')
        main_ylim_full = plot_config.get('main_ylim_full')
        ax.set_xlim(main_xlim_full)
        ax.set_ylim(main_ylim_full)
        
        # Inset shows the zoomed-in part
        axins_xlim, axins_ylim = inset_xlim, inset_ylim

    ax.grid(True, linestyle='--', alpha=0.6,  linewidth=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # --- Inset Creation ---
    if inset_rect and inset_xlim and inset_ylim:
        axins = ax.inset_axes(inset_rect)
        for method in plotted_methods:
            edcs = data['edcs'][method]
            if receiver_index < len(edcs):
                edc = edcs[receiver_index]
                time_axis = np.arange(len(edc)) / Fs
                linestyle = get_linestyle(method)
                color = get_color(method, name_map)
                axins.plot(time_axis, edc, linestyle=linestyle, linewidth=1, color=color) #linewidth of inset
                
        # Add reflection order lines to the inset plot
        # if reflection_times:
        #     plot_reflection_vertical_lines(axins, reflection_times)

        axins.set_xlim(axins_xlim)
        axins.set_ylim(axins_ylim)
        axins.grid(True, linestyle='--', alpha=0.6, linewidth=0.5) # linewidth of inset grid
        
        # Restore tick formatting for the inset for better readability
        axins.tick_params(axis='x', labelsize=8)
        axins.tick_params(axis='y', labelsize=8)

        if focus_on_early:
            axins.yaxis.set_major_locator(plt.MultipleLocator(20))
            axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        else:
            axins.yaxis.set_major_locator(plt.MultipleLocator(2))
            axins.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        if not focus_on_early:
            ax.indicate_inset_zoom(axins, edgecolor="black")

    # --- Legend and Layout (adopted from paper_figures_ned.py) ---
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=4, fontsize=8.5) # fontsize of legend
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    # --- Save Figure ---
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # --- FIGURE GENERATION SETUP ---
    PROCESS_ALL_FILES = True
    # specific_files_to_process = ["aes_room_spatial_edc_data.npz"]

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

    if PROCESS_ALL_FILES:
        # Filter out the reflection_times.npz file from the processing list
        files_to_run = [f for f in os.listdir(data_dir) if f.endswith('.npz') and f != 'reflection_times.npz']
    else:
        files_to_run = specific_files_to_process

    print(f"--- Starting figure generation for {len(files_to_run)} data file(s) ---")
    
    for filename in files_to_run:
        print(f"\nProcessing file: {filename}")
        data_path = os.path.join(data_dir, filename)

        try:
            simulation_data = load_data(data_path)
        except FileNotFoundError as e:
            print(f"  Error: {e}. Skipping.")
            continue

        room_params = simulation_data['room_params']
        if isinstance(room_params, np.ndarray): # Handle case where it's stored as a numpy array
            room_params = room_params.item()
            
        room_name = room_params.get('display_name', 'unknown_room')
        output_filename = f"fig_edc_inset_comparison_{room_name.lower().replace(' ', '_')}.png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Generating plot for room: '{room_name}'")
        
        # --- PLOT CONFIGURATION ---
        methods_to_exclude = []
        
        # Get the specific plot settings for this room from our central config
        room_plot_config = PLOT_CONFIG.get(room_name, {}).get('edc', {})
        
        # Get reflection times for the current room from the data loaded earlier
        reflection_times = all_reflection_times.get(room_name, {}).get('reflection_times')

        plot_edc_with_inset(
            data=simulation_data, 
            receiver_index=0, 
            output_path=output_path,
            excluded_methods=methods_to_exclude,
            focus_on_early=True, # You can still toggle this mode
            name_map=DISPLAY_NAME_MAP,
            plot_config=room_plot_config,
            reflection_times=reflection_times
        )

    print("\n--- All figures generated. ---")
    plt.show() 