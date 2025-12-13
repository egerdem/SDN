import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

from analysis.plotting_utils import load_data
import analysis.plot_room as pp

# --- Configuration ---
DATA_DIR = os.path.join(project_root, "results", "paper_data")
OUTPUT_DIR = os.path.join(project_root, "results", "edc_source_variations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Files representing different source positions
FILES_TO_PROCESS = [
    # "aes_quarter_center_source.npz", # Example filename, adjust if needed
    # "aes_room_spatial_edc_data_top_middle_source.npz",
    # ... User provided list in prompt ...
    
    # Using the filenames likely present based on previous context:
    "aes_quarter_center_source.npz",
    "aes_quarter_top_middle_source.npz",
    "aes_quarter_upper_right_source.npz",
    "aes_quarter_lower_left_source.npz",
    "aes_quarter_corner_sourcev3.npz"

    # "aes_FULLGRID_center_source.npz",
    # "aes_FULLGRID_top_middle_source.npz",
    # "aes_FULLGRID_upper_right_source.npz",
    # "aes_FULLGRID_lower_left_source.npz",
    # "aes_FULLGRID_corner_sourcev3.npz"
]

RECEIVER_INDICES = [0, 15] # Example indices common in small grids (81 pts)

# Configurable list of methods to plot (matches typical keys in .npz)
METHODS_TO_PLOT = [
    'SDN-Test1', 
    # 'SDN-Test_3', 'SDN-Test_2', 'SDN-Test2', 'SDN-Test3', 'SDN-Test4','SDN-Test5',
    # 'SDN-Test6', 'SDN-Test7', 'HO-SDN-N2', 'HO-SDN-N3'
]
CALCULATE_NED = False

import textwrap
from matplotlib.widgets import CheckButtons, Button
import analysis.analysis as an
import analysis.EchoDensity as ned

def custom_interactive_plot_paged(pages_payload, Fs):
    """
    Paged interactive plot.
    pages_payload: list of dicts {'rirs_dict': ..., 'coords': ..., 'idx': ...}
    """
    if not pages_payload:
        print("No pages to plot.")
        return

    num_pages = len(pages_payload)
    current_page = [0] # Mutable to be accessible in callbacks

    # ---------------- Setup Figure ----------------
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=2/72, h_pad=2/72, hspace=0.02, wspace=0.02)
    
    # GridSpec: [Rows: NED/EDC, Legend, RIR, Nav] 
    gs = plt.GridSpec(4, 3, height_ratios=[0.38, 0.12, 0.42, 0.08], width_ratios=[1, 1, 0.25])
    
    ax_ned = fig.add_subplot(gs[0, 0])
    ax_edc = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 0:2])
    ax_legend.axis('off')
    ax_rir = fig.add_subplot(gs[2, 0:2])
    ax_check = fig.add_subplot(gs[:3, 2]) # Checkbox spans top 3 rows
    
    # Navigation Buttons Area
    ax_prev = fig.add_subplot(gs[3, 0])
    ax_page_info = fig.add_subplot(gs[3, 1])
    ax_next = fig.add_subplot(gs[3, 2])
    ax_page_info.axis('off')

    # Global refs for objects on current page
    active_objects = {
        'lines': [], # list of all line artists
        'check': None,
        'legend': None
    }
    
    # Color Cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    base_colors = prop_cycle.by_key()['color']
    
    def render_page(p_idx):
        # Clear Axes
        ax_rir.clear()
        ax_edc.clear()
        ax_ned.clear()
        ax_check.clear()
        if hasattr(fig, 'shared_legend') and fig.shared_legend:
            fig.shared_legend.remove()
        
        # Get Data
        data = pages_payload[p_idx]
        rirs_dict = data['rirs_dict']
        coords = data['coords']
        rx_idx = data['idx']
        
        # Title
        fig.suptitle(f"Receiver {rx_idx} Analysis\nPosition: {coords}", fontsize=14, fontweight='bold')
        ax_page_info.text(0.5, 0.5, f"Page {p_idx+1} / {num_pages}", ha='center', va='center', fontsize=12)
        ax_page_info.axis('off') # Ensure off

        # Sort Labels
        sorted_labels = sorted(rirs_dict.keys())
        
        # Color Mapping (Consistent per page)
        source_color_map = {} 
        c_i = 0
        
        rir_lines = {}
        edc_lines = {}
        ned_lines = {}
        
        for label in sorted_labels:
            rir = rirs_dict[label]
            
            # Determine Color
            base_name = label.replace(" (Ref)", "").split("(")[0].strip()
            if base_name not in source_color_map:
                source_color_map[base_name] = base_colors[c_i % len(base_colors)]
                c_i += 1
            color = source_color_map[base_name]
            
            # Style
            if "(Ref)" in label:
                ls, alpha, lw, z = '-', 1.0, 2.0, 10
            else:
                ls, alpha, lw, z = '--', 0.8, 1.5, 5
            
            # Plot
            t = np.arange(len(rir)) / Fs
            l_rir, = ax_rir.plot(t, rir, label=label, color=color, ls=ls, alpha=alpha, lw=lw/2, zorder=z)
            
            edc, t_edc, _ = an.compute_edc(rir, Fs, label=label, plot=False)
            l_edc, = ax_edc.plot(t_edc, edc, label=label, color=color, ls=ls, alpha=alpha, lw=lw, zorder=z)
            
            if CALCULATE_NED:
                ned_prof = ned.echoDensityProfile(rir, fs=Fs)
                t_ned = np.arange(len(ned_prof)) / Fs
                l_ned, = ax_ned.plot(t_ned, ned_prof, label=label, color=color, ls=ls, alpha=alpha, lw=lw, zorder=z)
                ned_lines[label] = l_ned
            
            rir_lines[label] = l_rir
            edc_lines[label] = l_edc

        # Styling Axes
        ax_rir.set_title('Impulse Response'); ax_rir.grid(True, alpha=0.3); ax_rir.set_xlabel('Time (s)')
        ax_edc.set_title('Energy Decay Curve'); ax_edc.grid(True, alpha=0.3); ax_edc.set_ylim(-65, 5); ax_edc.set_xlabel('Time (s)')
        ax_ned.set_title('Normalized Echo Density'); ax_ned.grid(True, alpha=0.3); ax_ned.set_xlabel('Time (s)')
        
        # Checkboxes
        ax_check.set_title("Toggle Visibility", fontweight='bold')
        ax_check.axis('off')
        wrapped = [textwrap.fill(l, 25) for l in sorted_labels]
        check = CheckButtons(ax_check, wrapped, [True]*len(wrapped))
        active_objects['check'] = check
        
        # Legend Logic
        def update_legend():
            vis_handles = [l for l in rir_lines.values() if l.get_visible()]
            vis_labels = [l.get_label() for l in rir_lines.values() if l.get_visible()]
            if hasattr(fig, 'shared_legend') and fig.shared_legend:
                fig.shared_legend.remove()
            if vis_handles:
                ncol = min(len(vis_handles), 5) if len(vis_handles) > 0 else 1
                fig.shared_legend = ax_legend.legend(vis_handles, vis_labels, loc='center', ncol=ncol, frameon=False, fontsize='small')

        update_legend()
        
        # Layout lookup
        label_lookup = dict(zip(wrapped, sorted_labels))
        
        def on_check(lbl):
            real = label_lookup[lbl]
            for d in [rir_lines, edc_lines, ned_lines]:
                if real in d: d[real].set_visible(not d[real].get_visible())
            update_legend()
            fig.canvas.draw_idle()
            
        check.on_clicked(on_check)
        
        # Redraw
        fig.canvas.draw_idle()

    # Buttons
    btn_prev = Button(ax_prev, 'Previous Receiver')
    btn_next = Button(ax_next, 'Next Receiver')
    
    def on_prev(event):
        if current_page[0] > 0:
            current_page[0] -= 1
            render_page(current_page[0])
            
    def on_next(event):
        if current_page[0] < num_pages - 1:
            current_page[0] += 1
            render_page(current_page[0])
            
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    
    # Render Initial Page
    render_page(0)
    plt.show(block=True)


def analyze_variations():
    print("--- Starting EDC Source Variation Analysis ---")
    data_map = {}
    
    # 1. Load Data
    for filename in FILES_TO_PROCESS:
        path = os.path.join(DATA_DIR, filename)
        
        try:
            # Use standardized loader
            sim_data = load_data(path)
            
            # Extract clean source name
            name_clean = filename.replace("aes_quarter_", "").replace("aes_FULLGRID_", "").replace(".npz", "").replace("_", " ").title()
            
            # Check available methods in the loaded structure
            available_methods = list(sim_data['edcs'].keys())
            print(f"  Available keys in '{filename}': {available_methods}")
            
            # Find RIMPY (Reference) - flexible matching
            rimpy_name = "RIMPY-neg10"
            
            # Find Requested Methods
            loaded_methods = {}
            loaded_rirs = {}
            
            # Load RIMPY RIR if available
            rimpy_rir = None
            if rimpy_name in sim_data['rirs']:
                 rimpy_rir = sim_data['rirs'][rimpy_name]

            for method_key in METHODS_TO_PLOT:
                if method_key in sim_data['edcs']:
                    # Store EDC
                    data = sim_data['edcs'][method_key]
                    loaded_methods[method_key] = data
                    print(f"    FOUND '{method_key}': Shape={data.shape}, Dtype={data.dtype}")
                    if data.size > 0:
                        print(f"      Stats: Min={np.min(data):.2e}, Max={np.max(data):.2e}, Mean={np.mean(data):.2e}")
                        if np.any(np.isnan(data)): print("      WARNING: Contains NaNs")
                        if np.any(data < 0): print("      WARNING: Contains Negatives")
                    
                    # Store RIR if available
                    if method_key in sim_data['rirs']:
                        loaded_rirs[method_key] = sim_data['rirs'][method_key]
                else:
                    print(f"    NOT FOUND '{method_key}' in {available_methods}")

            entry = {
                'receiver_positions': sim_data['receiver_positions'],
                'rimpy': sim_data['edcs'].get(rimpy_name),
                'rimpy_rir': rimpy_rir,
                'methods': loaded_methods,
                'rirs': loaded_rirs,
                'room_params': sim_data.get('room_params'),
                'filename': filename
            }
            data_map[name_clean] = entry
            
            if entry['rimpy'] is not None:
                r_data = entry['rimpy']
                print(f"    RIMPY Stats: Min={np.min(r_data):.2e}, Max={np.max(r_data):.2e}")
            else:
                print("    RIMPY: None")
            
            print(f"Loaded {name_clean}: Receivers={len(sim_data['receiver_positions'])}, Methods={list(loaded_methods.keys())}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    # 2. Prepare Pages for Plotting
    if not data_map:
        print("No data loaded.")
        return

    pages_payload = []
    
    for rx_idx in RECEIVER_INDICES:
        print(f"\nCollecting data for Receiver Index {rx_idx}...")
        
        combined_rirs_for_plot = {}
        
        # We need the coordinates. Let's get them from the first available source file
        first_source = next(iter(data_map.values()))
        if rx_idx >= len(first_source['receiver_positions']):
            print(f"Index {rx_idx} out of bounds (max {len(first_source['receiver_positions'])-1}). Skipping.")
            continue
            
        rx_pos = first_source['receiver_positions'][rx_idx]
        if len(rx_pos) >= 3:
            rx_coords_str = f"({rx_pos[0]:.2f}, {rx_pos[1]:.2f}, {rx_pos[2]:.2f})"
        else:
            rx_coords_str = f"({rx_pos[0]:.2f}, {rx_pos[1]:.2f})"
        print(f"  Coordinates: {rx_coords_str}")
        
        # Collect all RIRs for this receiver from all sources
        for i, (source_name, loaded_data) in enumerate(data_map.items()):
            
            # Verify coordinates match (comparing common dimensions)
            this_pos = loaded_data['receiver_positions'][rx_idx]
            min_dim = min(len(rx_pos), len(this_pos))
            if not np.allclose(rx_pos[:min_dim], this_pos[:min_dim], atol=0.1):
                print(f"  WARNING: Receiver pos mismatch in {source_name}: {this_pos[:min_dim]} vs {rx_pos[:min_dim]}")
            
            # Add RIMPY RIR
            if loaded_data['rimpy_rir'] is not None:
                rir = loaded_data['rimpy_rir'][rx_idx]
                combined_rirs_for_plot[f"{source_name} (Ref)"] = rir

            # Add Method RIRs
            for m_name, m_rirs in loaded_data['rirs'].items():
                rir = m_rirs[rx_idx]
                label = f"{source_name}" if len(loaded_data['rirs']) == 1 else f"{source_name} ({m_name})"
                combined_rirs_for_plot[label] = rir
        
        if not combined_rirs_for_plot:
            print("  No RIRs found for this receiver index.")
            continue
            
        pages_payload.append({
            'idx': rx_idx,
            'coords': rx_coords_str,
            'rirs_dict': combined_rirs_for_plot
        })

    # 3. Launch Paged Interactive Plot
    print(f"\nPrepared {len(pages_payload)} pages. Launching Interactive Plot...")
    custom_interactive_plot_paged(pages_payload, Fs=44100)
    print("Plot closed.")

if __name__ == "__main__":
    analyze_variations()
