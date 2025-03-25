import librosa
import numpy as np
import os
import json
import h5py
from sdn_manager_load_sims import get_batch_manager, SDNExperiment, Room

from treble_tsdk.tsdk import TSDK, TSDKCredentials
tsdk = TSDK(TSDKCredentials.from_file("/Users/ege/Projects/SDN/TREBLE/tsdk.cred"))
from treble_tsdk.geometry.generator import GeometryGenerator
# The tsdk_namespace provides easy access to SDK object types.
from treble_tsdk import tsdk_namespace as treble
# The display_data module provides ways to display SDK object data as trees and tables.
from treble_tsdk import display_data as dd
from treble_tsdk.results import plot

"""my_projects = tsdk.list_my_projects()
# dd.as_table(my_projects)

project = my_projects[0]
simulations = project.get_simulations()
# destination_directory = "./results/treble/"
# project.download_results(destination_directory, rename_rule=treble.ResultRenameRule.by_label)

# Get the two different simulations
simulation_0 = simulations[0]  # First simulation
# simulation_1 = simulations[1]  # Second simulation

print(f"Simulation 0: {simulation_0.name}")

source_dir = "./results/treble/SDN-AES Center_Source" #there are 3 other source folders in this directory
# load the json file from source_dir, use python's json module to load the file
json_path = source_dir + "/simulation_info.json"
with open(json_path, 'r') as f:
    simulation_info = json.load(f)

# get receivers and sources from the simulation_info dictionary
receiver_info = simulation_info["receivers"]
# iterate over the receivers and get 'x', 'y', 'z' key value pairs, store them in a list
receiver_positions = {}

for receiver in receiver_info:
    receiver_position = [receiver["x"], receiver["y"], receiver["z"]]
    receiver_positions[receiver["label"]] = receiver_position

source_info = simulation_info["sources"]
source_positions = {}
for source in source_info:
    source_position = [source["x"], source["y"], source["z"]]
    source_positions[source["label"]] = source_position"""

#also log "name", "simulationType",  "layerMaterialAssignments"[0]."materialName"

#read h5.file inside the source_dir and get mono rir.
import h5py
h5_file_path = source_dir + "/Center_Source_Hybrid.h5"
# with h5py.File(h5_file_path, 'r') as f:
#     print("?")


def explore_h5_file(h5_file_path):
    """Get detailed information about the H5 file structure including data shapes and types."""
    with h5py.File(h5_file_path, 'r') as f:
        print("H5 File Structure:")
        
        # Check the main groups
        print("Main Groups:", list(f.keys()))
        
        # Print file-level attributes
        print("\nFile Attributes:")
        for key, val in f.attrs.items():
            print(f"  - {key}: {val}")
        
        # Look at mono_ir dataset
        print("\nmono_ir details:")
        for key in f['mono_ir'].keys():
            dataset = f['mono_ir'][key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            
            # Check if the dataset has attributes
            if len(dataset.attrs) > 0:
                print(f"    Attributes:")
                for attr_key, attr_val in dataset.attrs.items():
                    print(f"      - {attr_key}: {attr_val}")
        
        # Check if there's time data
        if 'time' in f:
            time_data = f['time']
            print("\nTime data:", time_data.shape, time_data.dtype)
            if len(time_data) > 1:
                print(f"  Time step: {time_data[1] - time_data[0]}")
            if len(time_data.attrs) > 0:
                print(f"  Time attributes:")
                for attr_key, attr_val in time_data.attrs.items():
                    print(f"    - {attr_key}: {attr_val}")


# print_h5_structure("./results/treble/SDN-AES Center_Source/Center_Source_Hybrid.h5")

# define and experiment dict including these info which is copatible to the one inside the Room objects' rooms dict in @sdn_manager_load_sims' batch_manager

#later
# Define the results directory paths
# results_0 = simulation_0.get_results_object(source_dir)
# results_0.plot()
# treble_ir_1 = results_0.get_mono_ir("Omni_source", "mono_receiver")

def map_receivers_to_mono_ir(simulation_info):
    """Create a mapping from receiver positions to their IDs in the H5 file."""
    receiver_map = {}
    
    for receiver in simulation_info["receivers"]:
        receiver_id = receiver["id"]
        position = (receiver["x"], receiver["y"], receiver["z"])
        label = receiver["label"]
        
        receiver_map[receiver_id] = {
            "position": [receiver["x"], receiver["y"], receiver["z"]],
            "label": label
        }
    
    return receiver_map

def load_treble_rirs(h5_file_path, receiver_map):
    """
    Load all Room Impulse Responses from the H5 file.
    
    Args:
        h5_file_path: Path to the H5 file
        receiver_map: Mapping from receiver IDs to position data
        
    Returns:
        dict: Mapping from receiver IDs to RIR data
    """
    rirs = {}
    
    with h5py.File(h5_file_path, 'r') as f:
        # Get sampling rate directly from file attributes
        fs_treble = f.attrs.get('sampling-rate', 32000)  # Default to 32000 if not found
        target_fs = 44100  # Target sampling rate for consistency with SDN/ISM
        
        # Get zero-padding information
        zero_pad_beginning = f.attrs.get('zero-pad-beginning', 0)
        zero_pad_end = f.attrs.get('zero-pad-end', 0)
        
        print(f"Treble sampling rate: {fs_treble} Hz, resampling to {target_fs} Hz")
        print(f"Zero padding: beginning={zero_pad_beginning}, end={zero_pad_end} samples")
        
        # Load all mono IRs
        for receiver_id in f['mono_ir'].keys():
            if receiver_id in receiver_map:
                # Get the original RIR
                original_rir = f['mono_ir'][receiver_id][:]
                
                # Remove zero-padding
                if zero_pad_beginning > 0 or zero_pad_end > 0:
                    if zero_pad_end > 0:
                        unpadded_rir = original_rir[zero_pad_beginning:-zero_pad_end]
                    else:
                        unpadded_rir = original_rir[zero_pad_beginning:]
                    print(f"Removed padding: original length={len(original_rir)}, unpadded length={len(unpadded_rir)}")
                else:
                    unpadded_rir = original_rir
                
                # Resample using librosa
                resampled_rir = librosa.resample(
                    y=unpadded_rir, 
                    orig_sr=fs_treble, 
                    target_sr=target_fs
                )
                
                # Normalize RIR to max amplitude of 1 after resampling
                if np.max(np.abs(resampled_rir)) > 0:
                    resampled_rir = resampled_rir / np.max(np.abs(resampled_rir))
                
                rirs[receiver_id] = {
                    "rir": resampled_rir,
                    "position": receiver_map[receiver_id]["position"],
                    "label": receiver_map[receiver_id]["label"],
                    "fs": target_fs,
                    "original_fs": fs_treble,
                    "zero_pad_beginning": zero_pad_beginning,
                    "zero_pad_end": zero_pad_end
                }
            else:
                print(f"Warning: Receiver ID {receiver_id} not found in simulation info")
    
    return rirs

def create_treble_experiments(simulation_info, source_info, rirs):
    """
    Create SDNExperiment objects from Treble simulation data.
    
    Args:
        simulation_info: The simulation info dictionary
        source_info: The source information dictionary
        rirs: Dict mapping from receiver IDs to RIR data
        
    Returns:
        list: List of SDNExperiment objects
    """
    experiments = []
    
    # Extract room parameters from simulation info
    room_params = {
        'width': 9.0,  # Default values, may need to extract from simulation geometry
        'depth': 7.0,
        'height': 4.0,
        'absorption': 0.2  # Extract from material properties if possible
    }
    
    # Try to extract material absorption
    if "layerMaterialAssignments" in simulation_info and simulation_info["layerMaterialAssignments"]:
        material_name = simulation_info["layerMaterialAssignments"][0].get("materialName", "")
        print("material name:", material_name)
        if "-" in material_name:
            try:
                # Extract absorption value from name like "SDN-AES-0.2"
                absorption_str = material_name.split("-")[-1]
                room_params['absorption'] = float(absorption_str)
            except (ValueError, IndexError):
                pass
    
    # Extract simulation parameters
    duration = simulation_info.get("impulseLengthSec", 2.0)
    sim_type = simulation_info.get("simulationType", "Hybrid")
    crossover = simulation_info.get("crossoverFrequency", 250)
    sim_settings = simulation_info.get("simulationSettings", {})

    # For each source in the simulation
    source_pos = [source_info["x"], source_info["y"], source_info["z"]]
    source_label = source_info["label"]
    
    # For each receiver with RIR data
    for receiver_id, rir_data in rirs.items():
        receiver_pos = rir_data["position"]
        receiver_label = rir_data["label"]
        fs = rir_data["fs"]  # Use the sampling rate from the RIR data (now 44100 after resampling)
        
        # Create config similar to SDN/ISM configs
        config = {
            'method': 'TREBLE',
            'label': f"TREBLE {sim_type}",
            'info': f"XO={crossover}Hz",
            'fs': fs,
            'duration': duration,
            'room_parameters': {
                'width': room_params['width'],
                'depth': room_params['depth'],
                'height': room_params['height'],
                'absorption': room_params['absorption'],
                'source x': source_pos[0],
                'source y': source_pos[1],
                'source z': source_pos[2],
                'mic x': receiver_pos[0],
                'mic y': receiver_pos[1],
                'mic z': receiver_pos[2]
            },
            'source': {
                'position': source_pos,
                'label': source_label
            },
            'receiver': {
                'position': receiver_pos,
                'label': receiver_label,
                'id': receiver_id
            },
            'treble_settings': {
                'simulation_type': sim_type,
                'simulation_settings': sim_settings,
                'crossover_frequency': crossover,
                'simulation_id': simulation_info.get('id', ''),
                'original_fs': rir_data.get("original_fs", 32000)
            }
        }
        
        # Create a unique experiment ID
        experiment_id = f"treble_{source_label}_{receiver_label}_{receiver_id[:8]}"
        
        # Create the experiment object with resampled RIR
        experiment = SDNExperiment(
            config=config,
            rir=rir_data["rir"],
            fs=fs,
            duration=duration,
            experiment_id=experiment_id
        )
        
        experiments.append(experiment)
    
    return experiments

def process_treble_source_directory(source_dir, batch_manager, room_name="room_aes"):
    """
    Process a Treble source directory and add experiments to the batch manager.
    
    Args:
        source_dir: Path to the Treble source directory
        batch_manager: The SDNExperimentManager instance
        room_name: Name of the room to add experiments to
    """
    # Find the simulation_info.json file
    sim_info_path = os.path.join(source_dir, "simulation_info.json")
    if not os.path.exists(sim_info_path):
        print(f"No simulation_info.json found in {source_dir}")
        return
    
    # Load simulation info
    with open(sim_info_path, 'r') as f:
        simulation_info = json.load(f)
    
    # Find the H5 file
    h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]
    if not h5_files:
        print(f"No H5 file found in {source_dir}")
        return
    
    h5_file_path = os.path.join(source_dir, h5_files[0])
    
    # Get source info (assumes one source per directory)
    source_info = simulation_info["sources"][0]
    
    # Create receiver ID to position mapping
    receiver_map = map_receivers_to_mono_ir(simulation_info)
    
    # Load RIRs from H5 file
    rirs = load_treble_rirs(h5_file_path, receiver_map)
    
    # Get or create room in batch manager
    room = None
    for existing_name, existing_room in batch_manager.rooms.items():
        if existing_name == room_name or existing_room.display_name == room_name:
            room = existing_room
            break
    
    if room is None:
        print(f"Room {room_name} not found in batch manager. Creating new room.")
        # Create basic room parameters
        room_params = {
            'width': 9.0,
            'depth': 7.0,
            'height': 4.0,
            'absorption': 0.2,
            'room_name': room_name
        }

        # Try to extract material absorption
        if "layerMaterialAssignments" in simulation_info and simulation_info["layerMaterialAssignments"]:
            material_name = simulation_info["layerMaterialAssignments"][0].get("materialName", "")
            if "-" in material_name:
                try:
                    # Extract absorption value from name like "SDN-AES-0.2"
                    absorption_str = material_name.split("-")[-1]
                    room_params['absorption'] = float(absorption_str)
                except (ValueError, IndexError):
                    pass

        # Create the room
        room = Room(room_name, room_params)
        batch_manager.rooms[room_name] = room
    
    # Create and add experiments
    experiments = create_treble_experiments(simulation_info, source_info, rirs)
    
    # Add experiments to room
    for experiment in experiments:
        room.add_experiment(experiment)
    
    print(f"Added {len(experiments)} Treble experiments from {source_dir}")
    return experiments

def integrate_treble_results(treble_base_dir, batch_manager):
    """
    Process all Treble results in the base directory and add them to the batch manager.
    
    Args:
        treble_base_dir: Path to the base directory containing Treble results
        batch_manager: The SDNExperimentManager instance
    """
    if not os.path.exists(treble_base_dir):
        print(f"Treble base directory {treble_base_dir} not found")
        return
    
    # Process each source directory
    for item in os.listdir(treble_base_dir):
        source_dir = os.path.join(treble_base_dir, item)
        if os.path.isdir(source_dir):
            print(f"Processing Treble source directory: {item}")
            process_treble_source_directory(source_dir, batch_manager)

# Main function
if __name__ == "__main__":
    # First, explore an H5 file to understand its structure
    source_dir = "./results/treble/SDN-AES Center_Source"
    h5_file_path = os.path.join(source_dir, "Center_Source_Hybrid.h5")
    explore_h5_file(h5_file_path)
    
    # Load the batch manager
    batch_manager = get_batch_manager()
    
    # Process all Treble source directories
    treble_base_dir = "./results/treble"
    integrate_treble_results(treble_base_dir, batch_manager)
    
    # Print summary
    total_exps = sum(len(room.experiments) for room in batch_manager.rooms.values())
    print(f"\nTotal experiments loaded: {total_exps}")
    for room_name, room in batch_manager.rooms.items():
        print(f"Room {room_name}: {len(room.experiments)} experiments, {len(room.source_mic_pairs)} source-mic pairs")
    
    # Launch visualization
    try:
        from sdn_experiment_visualizer import SDNExperimentVisualizer
        visualizer = SDNExperimentVisualizer(batch_manager)
        visualizer.show(port=8085)
    except ImportError:
        print("SDNExperimentVisualizer not available, skipping visualization")