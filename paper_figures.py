import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

# Import shared functions and variables from our new utility module
from plotting_utils import (
    load_data, 
    get_display_name, 
    get_linestyle, 
    PLOT_CONFIG
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
    
    fig, ax = plt.subplots(figsize=[6, 5])
    
    print(f"Plotting EDCs for receiver position {receiver_index}...")
    
    # --- PLOTTING LOOPS ---
    for method in name_map.keys():
        if method in excluded_methods or method not in data['edcs']:
            continue
            
        edcs = data['edcs'][method]
        if receiver_index < len(edcs):
            edc = edcs[receiver_index]
            time_axis = np.arange(len(edc)) / Fs
            label = get_display_name(method, method_configs, name_map)
            linestyle = get_linestyle(method)
            ax.plot(time_axis, edc, label=label, linestyle=linestyle)

    # Add reflection order lines to the main plot
    if reflection_times:
        _plot_reflection_vertical_lines(ax, reflection_times)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (dB)")
    
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

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # --- Inset Creation ---
    axins = ax.inset_axes(inset_rect)
    for method in name_map.keys():
        if method in excluded_methods or method not in data['edcs']:
            continue
        edcs = data['edcs'][method]
        if receiver_index < len(edcs):
            edc = edcs[receiver_index]
            time_axis = np.arange(len(edc)) / Fs
            linestyle = get_linestyle(method)
            axins.plot(time_axis, edc, linestyle=linestyle)
            
    # Add reflection order lines to the inset plot
    if reflection_times:
        _plot_reflection_vertical_lines(axins, reflection_times)

    axins.set_xlim(axins_xlim)
    axins.set_ylim(axins_ylim)
    axins.grid(True, linestyle='--', alpha=0.6)
    
    # Restore tick formatting for the inset for better readability
    axins.tick_params(axis='x', labelsize=8)
    axins.tick_params(axis='y', labelsize=8)

    if focus_on_early:
        # When main plot is focused, inset shows the full view.
        # Set y-ticks to be every 20 dB for a cleaner look.
        axins.yaxis.set_major_locator(plt.MultipleLocator(20))
        axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        # When main plot is the full view, inset is zoomed-in.
        # Use a smaller tick interval for the focused view.
        axins.yaxis.set_major_locator(plt.MultipleLocator(2))
        axins.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    if not focus_on_early:
        ax.indicate_inset_zoom(axins, edgecolor="black")

    # --- Legend Placement ---
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3, fontsize='small')

    # --- Save Figure ---
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # --- FIGURE GENERATION SETUP ---
    PROCESS_ALL_FILES = True
    specific_files_to_process = ["aes_room_spatial_edc_data.npz"]

    data_dir = "results/paper_data"
    output_dir = "results/paper_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    if PROCESS_ALL_FILES:
        files_to_run = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
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

        room_name = simulation_data['room_params'].get('display_name', 'unknown_room')
        output_filename = f"fig_edc_inset_comparison_{room_name.lower().replace(' ', '_')}.png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Generating plot for room: '{room_name}'")
        
        # --- PLOT CONFIGURATION ---
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
        
        methods_to_exclude = []
        
        # Get the specific plot settings for this room from our central config
        room_plot_config = PLOT_CONFIG.get(room_name, {}).get('edc', {})
        
        # Extract reflection times, making sure it's a dictionary
        reflection_times_data = simulation_data.get('reflection_times')
        if reflection_times_data is not None:
            reflection_times = reflection_times_data.item()
        else:
            reflection_times = None

        plot_edc_with_inset(
            data=simulation_data, 
            receiver_index=0, 
            output_path=output_path,
            excluded_methods=methods_to_exclude,
            focus_on_early=True, # You can still toggle this mode
            name_map=display_name_and_order,
            plot_config=room_plot_config,
            reflection_times=reflection_times
        )

    print("\n--- All figures generated. ---")
    plt.show() 