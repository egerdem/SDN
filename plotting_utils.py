import numpy as np
import os
from typing import Dict

# --- Room Definitions ---
# Central place to define room parameters for consistency.
ROOM_AES = { 'display_name': 'AES Room' }
ROOM_JOURNAL = { 'display_name': 'Journal Room' }
ROOM_WASPAA = { 'display_name': 'WASPAA Room' }

# --- Plotting Style and Configuration ---
# This dictionary holds all the specific parameters for plotting,
# allowing for easy adjustments for each room without changing the script logic.
PLOT_CONFIG = {
    'AES Room': {
        'edc': {
            'main_xlim_focused': (0.005, 0.07),
            'main_ylim_focused': (-8, 0.2),
            'main_xlim_full': (0, 1.1),
            'main_ylim_full': (-62, 5),
            'inset_rect': [0.07, 0.07, 0.4, 0.3],
            'inset_xlim': (0.012, 0.03),
            'inset_ylim': (-5, 0.4)
        },
        'ned': {
            'main_xlim': (0, 0.2),
            'main_ylim': (0, 1.3),
            'inset_rect': [0.05, 0.51, 0.2, 0.45],  # x, y, width, height
            'inset_xlim': (0.025, 0.052),
            'inset_ylim': (0.0, 0.6)
        }
    },
    'Journal Room': {
        'edc': {
            'main_xlim_focused': (0.00, 0.07),
            'main_ylim_focused': (-8, 0.2),
            'main_xlim_full': (0, 1.1),
            'main_ylim_full': (-62, 5),
            'inset_rect': [0.07, 0.07, 0.4, 0.3],
            'inset_xlim': (0, 1.2),
            'inset_ylim': (-62, 5)
        },
        'ned': {
            'main_xlim': (0, 0.2),
            'main_ylim': (0, 1.3),
            'inset_rect': [0.50, 0.07, 0.2, 0.45], # x, y, width, height
            'inset_xlim': (0.0, 0.025),
            'inset_ylim': (0, 0.3)
        }
    },
    'WASPAA Room': {
        'edc': {
            'main_xlim_focused': (0.01, 0.07),
            'main_ylim_focused': (-5, 0.2),
            'main_xlim_full': (0, 1.8),
            'main_ylim_full': (-62, 5),
            'inset_rect': [0.07, 0.07, 0.4, 0.3],
            'inset_xlim': (0.012, 0.03),
            'inset_ylim': (-5, 0.4)
        },
        'ned': {
            'main_xlim': (0, 0.2),
            'main_ylim': (0, 1.3),
            'inset_rect': [0.50, 0.07, 0.2, 0.45], # x, y, width, height
            'inset_xlim': (0.015, 0.035),
            'inset_ylim': (0, 0.25)
        }
    },
}

# --- Display Name Mapping ---
# Central source for publication-ready names for different simulation methods.
DISPLAY_NAME_MAP = {
    'ISM-pra': 'ISM (PRA)',
    'ISM-pra-rand10' : {'name': 'ISM (PRA-10)', 'color': 'tab:blue'},
    'RIMPY-neg': 'ISM (rimpy-neg)',
    'RIMPY-neg10' : {'name':'ISM (neg10)', 'color': 'tab:blue'},
    'ISM (rimpy-neg10)' : {'name': 'ISM (rimpy-neg10)', 'color': 'tab:blue'},
    'ISM (rimpy-neg)' : {'name': 'ISM (rimpy-neg)', 'color': 'tab:blue'},
    'ISM (rimpy-pos)' : {'name': 'ISM (rimpy-pos)', 'color': 'tab:blue'},
    'SDN-Test1': {'name': 'SDN Original (c=1)', 'color': 'red'},
    'SDN-Test1n': {'name':'SDN Original, no loss', 'color': 'red'},
    'SDN-Test_3': 'SW-SDN (c=-3)',
    'SDN-Test_2': 'SW-SDN (c=-2)',
    'SDN-Test_1': 'SW-SDN (c=-1)',
    'SDN-Test_0': 'SW-SDN (c=0)',
    'SDN-Test2': 'SW-SDN (c=2)',
    'SDN-Test3': 'SW-SDN (c=3)',
    'SDN-Test4': 'SW-SDN (c=4)',
    'SDN-Test5': 'SW-SDN (c=5)',
    'SDN-Test6': 'SW-SDN (c=6)',
    'SDN-Test7': 'SW-SDN (c=7)',
    'SDN-Test_3r': 'SW-SDN-R (c=-3)',
    'SDN-Test_2r': 'SW-SDN-R (c=-2)',
    'SDN-Test_1r': 'SW-SDN-R (c=-1)',
    'SDN-Test_0r': 'SW-SDN-R (c=0)',
    'SDN-Test2r': 'SW-SDN-R (c=2)',
    'SDN-Test3r': 'SW-SDN-R (c=3)',
    'SDN-Test4r': 'SW-SDN-R (c=4)',
    'SDN-Test5r': 'SW-SDN-R (c=5)',
    'SDN-Test6r': 'SW-SDN-R (c=6)',
    'SDN-Test7r': 'SW-SDN-R (c=7)',
    'HO-SDN-N2': 'HO-SDN (N=2)',
    'HO-SDN-N3': 'HO-SDN (N=3)',
}

def get_display_name(method_key: str, method_configs: dict, name_map: dict) -> str:
    """
    Generate a publication-ready display name for a given method.
    This provides a single place to control legend labels.
    """
    config = method_configs.get(method_key, {})
    info = config.get('info', method_key) # Default to key if no info
    
    entry = name_map.get(method_key, info)
    
    if isinstance(entry, dict):
        return entry.get('name', info)
        
    return entry

def get_color(method_key: str, name_map: dict) -> str | None:
    """
    Get a specific color for a method if defined in the style map.
    """
    entry = name_map.get(method_key)
    if isinstance(entry, dict):
        return entry.get('color')
    return None

def get_linestyle(method_key: str) -> str:
    """
    Assign a specific linestyle based on the method type for the plot.
    This provides a single place to control line styles.
    """
    if method_key.startswith('HO-SDN'):
        return ':'  # Dotted line for all HO-SDN methods
    elif method_key == 'ISM':
        return '--' # Dashed line for ISM
    elif method_key.startswith('SDN-'):
        return '-'  # Solid line for the user's SW-SDN approach
    else:
        return '-.' # A default style for any other method

def load_data(data_path: str) -> dict:
    """Load the pre-calculated simulation data from an .npz file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. "
                              f"Please run generate_paper_data.py first.")
    
    data = np.load(data_path, allow_pickle=True)
    
    # Extract data into a more usable dictionary format
    results = {
        'edcs': {},
        'neds': {},
        'rirs': {},
        'receiver_positions': data['receiver_positions'],
        'source_pos': data['source_pos'],
        'Fs': int(data['Fs']),
        'duration': float(data['duration']),
        'method_configs': data['method_configs'][0],
        'room_params': data['room_params'][0]
    }
    
    for key in data.keys():
        if key.startswith('rirs_'):
            method = key.replace('rirs_', '')
            results['rirs'][method] = data[key]
        elif key.startswith('edcs_'):
            method = key.replace('edcs_', '')
            results['edcs'][method] = data[key]
        elif key.startswith('neds_'):
            method = key.replace('neds_', '')
            results['neds'][method] = data[key]
            
    return results

def plot_reflection_vertical_lines(ax, reflection_times):
    """Helper to draw vertical lines for reflection arrival times on a given axis."""
    if not reflection_times:
        return

    colors = ['#d62728', '#2ca02c', '#1f77b4']  # More distinct colors
    order_map = {
        'first_order': 1,
        'second_order': 2,
        'third_order': 3
    }
    
    # Check if there are any existing lines to decide on legend entries
    has_existing_labels = bool(ax.get_legend_handles_labels()[1])
    
    for order_name, time in reflection_times.items():
        if time is not None:
            order_num = order_map.get(order_name)
            if order_num is not None and order_num <= len(colors):
                color = colors[order_num - 1]
                color = "red"
                
                ax.axvline(x=time, color=color, linestyle='-', alpha=0.3, linewidth=1,)
                
                # Add text annotation just above the x-axis
                ax.text(time, ax.get_ylim()[-1], "", # Added space for padding
                        color=color, alpha=0.9,
                        ha='center', va='bottom', backgroundcolor=(1,1,1,0.6),
                        fontsize='x-small',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1))