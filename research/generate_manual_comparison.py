import numpy as np
import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

import experiment_configs as exp_config
from generate_paper_data import calculate_and_save_data

def generate_manual_comparison():
    print("--- GENERATING MANUAL COMPARISON DATA ---")
    
    # 1. Setup Simulation Parameters
    active_room = exp_config.active_room
    print(f"Room: {active_room['display_name']}")
    
    # Custom Receiver
    receiver_positions = [(6.5, 3.0, 2.0)]
    
    # Custom Sources
    sources = [
        ("top_middle", (4.5, 6.0, 2.0)),
        ("upper_right", (8.5, 6.0, 2.0))
    ]
    
    # Methods to Run (Ref + SDN C=1)
    # We'll use the standard configs but filter for what we need
    # Assuming 'SDN-Test1' is the standard C=1, or we define it explicitly if needed.
    # We'll grab all enabled ones from exp_config, or force specific ones.
    
    # Let's verify 'SDN-Test1' configuration or similar.
    # Typically user uses 'SDN-Test1' in the plotting script.
    
    method_configs = {}
    method_configs.update(exp_config.ism_methods)
    method_configs.update(exp_config.sdn_tests)
    
    # Force enable specific methods for this comparison
    if 'RIMPY-neg10' in method_configs:
        method_configs['RIMPY-neg10']['enabled'] = True
        print("Enabled RIMPY-neg10")
    
    if 'SDN-Test1' in method_configs:
        method_configs['SDN-Test1']['enabled'] = True
        print("Enabled SDN-Test1")

    # Disable 'SDN-Test1_spec' if it was accidentally enabled and not needed, 
    # but the logs showed it ran. If user only asked for "original sdn c=1", Test1 is it.

    
    Fs = exp_config.Fs
    duration = exp_config.duration
    
    output_dir = os.path.join(project_root, "results", "paper_data")
    
    for name, pos in sources:
        print(f"\nProcessing Source: {name} at {pos}")
        
        filename = f"aes_manual_{name}_source.npz"
        output_path = os.path.join(output_dir, filename)
        
        calculate_and_save_data(
            room_params=active_room,
            source_pos=pos,
            receiver_positions=receiver_positions,
            duration=duration,
            Fs=Fs,
            method_configs=method_configs,
            output_path=output_path,
            update_mode=True # Load existing wrapper? No, create new. But True handles creation too.
        )
        
    print("\n--- Manual Generation Complete ---")

if __name__ == "__main__":
    generate_manual_comparison()
