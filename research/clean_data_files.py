import numpy as np
import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Define the files to process (same as in generate_paper_data.py)
FILES_TO_PROCESS = [
    # "aes_room_spatial_edc_data_center_source.npz",
    "aes_room_spatial_edc_data_top_middle_source.npz",
    # "aes_room_spatial_edc_data_upper_right_source.npz",
    # "aes_room_spatial_edc_data_lower_left_source.npz",
]

# Define methods to KEEP (all others will be removed)
METHODS_TO_KEEP = [
    'RIMPY-neg10',
    # Add others here if needed, e.g., 'ISM'
]

def clean_data_files():
    # Get absolute path to results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(project_root, "results", "paper_data")

    print(f"--- Cleaning Data Files in {data_dir} ---")
    print(f"Keeping methods: {METHODS_TO_KEEP} (and any containing 'rimpy')")
    print(f"Processing files: {FILES_TO_PROCESS}")
    print("-" * 60)

    for filename in FILES_TO_PROCESS:
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {filename} - Skipping")
            continue

        print(f"\nProcessing: {filename}")
        
        try:
            print(f"  NumPy Version: {np.__version__}")
            # Load existing data
            with np.load(file_path, allow_pickle=True) as loaded_data:
                print(f"  Keys in file: {list(loaded_data.files)}")
                
                # Create a dictionary to store the data we want to keep
                new_data = {}
                
                # Copy metadata fields
                metadata_keys = ['receiver_positions', 'room_params', 'source_pos', 'Fs', 'duration', 'method_configs']
                for key in metadata_keys:
                    if key in loaded_data:
                        new_data[key] = loaded_data[key]
                
                # Iterate over all keys in the file
                removed_count = 0
                kept_count = 0
                
                for key in loaded_data.files:
                    if key in metadata_keys:
                        continue
                        
                    # Check if it's a method-specific key (rirs_, edcs_, neds_, rt60s_)
                    if any(key.startswith(prefix) for prefix in ['rirs_', 'edcs_', 'neds_', 'rt60s_']):
                        # Extract method name
                        # e.g., rirs_SDN-Test2.998 -> SDN-Test2.998
                        prefix = key.split('_')[0] + '_'
                        method_name = key[len(prefix):]
                        
                        # Check if method name contains 'rimpy' (case-insensitive)
                        if 'rimpy' in method_name.lower():
                            new_data[key] = loaded_data[key]
                            kept_count += 1
                            # print(f"  Keeping: {key}")
                        else:
                            removed_count += 1
                            print(f"  Removing: {key}")
                    else:
                        # Keep unknown keys just in case, or decide to remove them
                        new_data[key] = loaded_data[key]
                        print(f"  Keeping unknown key: {key}")

                # Save the cleaned data back to the same file
                np.savez(file_path, **new_data)
                print(f"  Done. Removed {removed_count} items, Kept {kept_count} items.")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

if __name__ == "__main__":
    clean_data_files()
