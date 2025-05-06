import os
import json
import numpy as np
import h5py
import librosa


# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Define a simplified version of the SDNExperiment class for Treble data
class TrebleExperiment:
    """Simple class to store Treble experiment data for later conversion to SDNExperiment"""

    def __init__(self, config, rir, fs=44100, duration=2.0, experiment_id=None):
        self.config = config
        self.rir = rir
        self.fs = fs
        self.duration = duration
        self.experiment_id = experiment_id or f"treble_{config.get('source', {}).get('label', '')}_{config.get('receiver', {}).get('label', '')}"


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
    Create TrebleExperiment objects from Treble simulation data.

    Args:
        simulation_info: The simulation info dictionary
        source_info: The source information dictionary
        rirs: Dict mapping from receiver IDs to RIR data

    Returns:
        list: List of TrebleExperiment objects
    """
    experiments = []

    # Extract room parameters from simulation info
    room_params = {
        'width': 9.0,  # Default values, may need to extract from simulation geometry
        'depth': 7.0,
        'height': 4.0,
    }

    # Extract absorption from material assignments
    if "layerMaterialAssignments" in simulation_info and simulation_info["layerMaterialAssignments"]:
        # Take the first material since all are the same
        material_name = simulation_info["layerMaterialAssignments"][0].get("materialName", "")


    # Extract simulation parameters
    duration = simulation_info.get("impulseLengthSec")
    sim_type = simulation_info.get("simulationType", "empty")
    sim_settings = simulation_info.get("simulationSettings", {})
    crossover = simulation_info.get("crossoverFrequency", 250)
    
    # Extract ISM order if available
    ism_order = sim_settings.get("gaSettings", {}).get("ISMOrder", "")
    
    # Create more descriptive label components
    method_str = "TRE"
    label_str = f"{sim_type}"
    if ism_order:
        label_str += f" ISM{ism_order}"
    
    # Add absorption to info if available
    # info_str = f"{material_name} , XO={crossover}Hz"
    info_str = f"{material_name}"

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
            'method': method_str,
            'label': label_str,
            'info': info_str,
            'fs': fs,
            'duration': duration,
            'room_parameters': {
                'width': room_params['width'],
                'depth': room_params['depth'],
                'height': room_params['height'],
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
                'original_fs': rir_data.get("original_fs")
            }
        }

        # Only add absorption if we have it
        if 'absorption' in room_params:
            config['room_parameters']['absorption'] = room_params['absorption']

        # Create a unique experiment ID
        experiment_id = f"treble_{source_label}_{receiver_label}_{receiver_id[:8]}"

        # Create the experiment object with resampled RIR
        experiment = TrebleExperiment(
            config=config,
            rir=rir_data["rir"],
            fs=fs,
            duration=duration,
            experiment_id=experiment_id
        )

        experiments.append(experiment)

    return experiments


def save_treble_experiments(treble_base_dir, experiments_by_source):
    """
    Save Treble experiments to disk for later loading by SDN batch manager.

    Args:
        treble_base_dir: Path to the base directory to save results
        experiments_by_source: Dict mapping from source label to list of experiments
    """
    output_dir = os.path.join(treble_base_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving processed Treble data to {output_dir}")

    for source_label, experiments in experiments_by_source.items():
        source_dir = os.path.join(output_dir, source_label)
        os.makedirs(source_dir, exist_ok=True)

        # Save room parameters from the first experiment (should be the same for all)
        room_params = experiments[0].config.get('room_parameters', {})
        with open(os.path.join(source_dir, 'room_info.json'), 'w') as f:
            json.dump({
                'name': 'room_aes',
                'display_name': 'room_aes',
                'parameters': room_params
            }, f, indent=2, cls=NumpyEncoder)  # Use NumPy encoder

        # Save each experiment
        for experiment in experiments:
            # Save metadata
            with open(os.path.join(source_dir, f"{experiment.experiment_id}.json"), 'w') as f:
                json.dump({
                    'experiment_id': experiment.experiment_id,
                    'config': experiment.config,
                    'fs': experiment.fs,
                    'duration': experiment.duration
                }, f, indent=2, cls=NumpyEncoder)  # Use NumPy encoder

            # Save RIR
            np.save(os.path.join(source_dir, f"{experiment.experiment_id}.npy"), experiment.rir)

        print(f"Saved {len(experiments)} experiments for source {source_label}")


def process_treble_source_directory(source_dir):
    """
    Process a Treble source directory and extract experiment data.

    Args:
        source_dir: Path to the Treble source directory

    Returns:
        tuple: (source_label, list of TrebleExperiment objects, source_folder_name)
    """
    # Get the folder name
    source_folder_name = os.path.basename(source_dir)
    
    # Find the simulation_info.json file
    sim_info_path = os.path.join(source_dir, "simulation_info.json")
    if not os.path.exists(sim_info_path):
        print(f"No simulation_info.json found in {source_dir}")
        return None, [], None

    # Load simulation info
    with open(sim_info_path, 'r') as f:
        simulation_info = json.load(f)

    # Find the H5 file
    h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]
    if not h5_files:
        print(f"No H5 file found in {source_dir}")
        return None, [], None

    h5_file_path = os.path.join(source_dir, h5_files[0])

    # Get source info (assumes one source per directory)
    source_info = simulation_info["sources"][0]
    source_label = source_info["label"]

    # Create receiver ID to position mapping
    receiver_map = map_receivers_to_mono_ir(simulation_info)

    # Load RIRs from H5 file
    rirs = load_treble_rirs(h5_file_path, receiver_map)

    # Create experiments
    experiments = create_treble_experiments(simulation_info, source_info, rirs)

    print(f"Processed {len(experiments)} Treble experiments from {source_dir}")

    return source_label, experiments, source_folder_name


def process_all_treble_sources(project_dir, is_singular=False):
    """
    Process all Treble results in the base directory, handling both direct and grouped experiments.
    
    Args:
        project_dir: Path to the base directory containing Treble results
        is_singular: If True, use folder names as keys
    """
    if not os.path.exists(project_dir):
        print(f"Treble base directory {project_dir} not found")
        return {}, {}

    experiments_by_source = {}
    folder_names_by_source = {}

    # Process each directory in the project folder
    for item in os.listdir(project_dir):
        if item.startswith('.') or item == "processed":
            continue

        item_path = os.path.join(project_dir, item)
        if not os.path.isdir(item_path):
            continue

        # Check if this is a grouped experiment set
        if any(f.endswith('.h5') for f in os.listdir(item_path)):
            # This is a direct result folder
            print(f"Processing direct result directory: {item}")
            source_label, experiments, folder_name = process_treble_source_directory(item_path)
            if source_label and experiments:
                key = folder_name if is_singular else source_label
                experiments_by_source[key] = experiments
                folder_names_by_source[key] = folder_name
        else:
            # This is a grouped experiment set
            print(f"Processing grouped experiment set: {item}")
            # Process each subdirectory in the group
            for subdir in os.listdir(item_path):
                subdir_path = os.path.join(item_path, subdir)
                if not os.path.isdir(subdir_path) or subdir.startswith('.'):
                    continue
                
                print(f"  Processing sub-experiment: {subdir}")
                source_label, experiments, folder_name = process_treble_source_directory(subdir_path)
                if source_label and experiments:
                    # Use parent folder name + subfolder name as the unique identifier
                    unique_key = f"{item}_{subdir}"
                    experiments_by_source[unique_key] = experiments
                    folder_names_by_source[unique_key] = unique_key

    return experiments_by_source, folder_names_by_source


def save_treble_experiments_matching_sdn_structure(treble_base_dir, experiments_by_source, PROJECT_NAME, room_name):
    """
    Save Treble experiments to match the SDN/ISM directory structure.

    Structure:
    {results_dir}/rooms/{room_name}/{source_label}/{method}/{param_set}/config.json
    {results_dir}/rooms/{room_name}/{source_label}/{method}/{param_set}/rirs.npy
    """
    # Base directory for rooms
    rooms_dir = os.path.join(treble_base_dir, "rooms")
    os.makedirs(rooms_dir, exist_ok=True)

    for source_label, experiments in experiments_by_source.items():
        # Get a sample experiment for room parameters
        sample_exp = experiments[0]
        room_params = sample_exp.config.get('room_parameters', {})

        # Save room info
        room_dir = os.path.join(rooms_dir, room_name)
        os.makedirs(room_dir, exist_ok=True)
        with open(os.path.join(room_dir, 'room_info.json'), 'w') as f:
            json.dump({
                'name': room_name,
                'display_name': room_name,
                'parameters': room_params
            }, f, indent=2, cls=NumpyEncoder)

        # Create source directory
        source_dir = os.path.join(room_dir, source_label)
        os.makedirs(source_dir, exist_ok=True)

        # Group experiments by sim type (for param_set)
        experiments_by_type = {}
        for exp in experiments:
            sim_type = exp.config.get('treble_settings', {}).get('simulation_type', 'hybrid')
            if sim_type not in experiments_by_type:
                experiments_by_type[sim_type] = []
            experiments_by_type[sim_type].append(exp)

        # Save each simulation type as a param_set
        for sim_type, exps in experiments_by_type.items():
            method_dir = os.path.join(source_dir, "TREBLE")
            os.makedirs(method_dir, exist_ok=True)

            param_set = sim_type.lower()
            # prepend PROJECT_NAME to the param_set string
            param_set = f"{PROJECT_NAME}_{param_set}"
            param_dir = os.path.join(method_dir, param_set)
            os.makedirs(param_dir, exist_ok=True)

            # Prepare config with receivers
            config = exps[0].config.copy()
            receivers = []
            rirs = []

            # Add each experiment as a receiver
            for exp in exps:
                # Extract receiver info
                receiver_info = exp.config.get('receiver', {})
                receiver_info['experiment_id'] = exp.experiment_id
                receivers.append(receiver_info)

                # Add RIR to array
                rirs.append(exp.rir)

            # Set the receivers list in config
            config['receivers'] = receivers

            # Save config
            with open(os.path.join(param_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2, cls=NumpyEncoder)

            # Save all RIRs in a single file
            rirs_array = np.array(rirs)
            np.save(os.path.join(param_dir, 'rirs.npy'), rirs_array)


def save_treble_experiments_for_singulars(global_results_dir, experiments_by_source, room_name, source_folder_name):
    """
    Save Treble experiments for singular cases with a flattened structure.
    All experiments are saved directly in the room_singulars directory with unique names.
    Uses the original experiment folder name as the unique identifier.
    
    Args:
        global_results_dir: Base results directory
        experiments_by_source: Dictionary of experiments grouped by source
        room_name: Name of the room
        source_folder_name: Original folder name of the experiment (e.g., 'journal_single loc_abs10_ism12_1sec')
    """
    # Base directory for singular rooms
    rooms_dir = os.path.join(global_results_dir, "room_singulars", "PROCESSED EXP_saved ")
    os.makedirs(rooms_dir, exist_ok=True)

    for source_label, experiments in experiments_by_source.items():
        # Get a sample experiment for room parameters
        sample_exp = experiments[0]
        room_params = sample_exp.config.get('room_parameters', {})

        # Save room info if it doesn't exist
        room_info_path = os.path.join(rooms_dir, 'room_info.json')
        if not os.path.exists(room_info_path):
            with open(room_info_path, 'w') as f:
                json.dump({
                    'name': room_name,
                    'display_name': room_name,
                    'parameters': room_params
                }, f, indent=2, cls=NumpyEncoder)

        # Use the source folder name as the unique identifier
        unique_id = source_folder_name

        # Prepare config with receivers
        config = experiments[0].config.copy()
        receivers = []
        rirs = []

        # Add each experiment as a receiver
        for exp in experiments:
            receiver_info = exp.config.get('receiver', {})
            receiver_info['experiment_id'] = f"{unique_id}_{exp.experiment_id}"
            receivers.append(receiver_info)
            rirs.append(exp.rir)

        # Set the receivers list in config
        config['receivers'] = receivers

        # Save config with unique name based on folder name
        config_path = os.path.join(rooms_dir, f'config_{unique_id}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)

        # Save RIRs with unique name based on folder name
        rirs_array = np.array(rirs)
        rirs_path = os.path.join(rooms_dir, f'rirs_{unique_id}.npy')
        np.save(rirs_path, rirs_array)


# Main function
if __name__ == "__main__":
    import sys
    import os
    
    # Configuration flags
    IS_SINGULAR = False  # Set to False for batch processing
    PREPROCESS_TREBLE_RESULTS = True

    # Directory setup
    PROJECT_ROOT = "./results"
    
    if IS_SINGULAR:
        PROJECT_NAME = "single_experiments"
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, "room_singulars")
        TREBLE_BASE_DIR = os.path.join(PROJECT_ROOT, "treble", "single_experiments")  # Direct path
        PROJECT_DIR = TREBLE_BASE_DIR  # Use TREBLE_BASE_DIR directly
    else:
        # PROJECT_NAME = "aes_abs20"  # Original treble batch results folder
        PROJECT_NAME = "aes_hybrid_ism12_abs20"  # New batch results folder
        PROJECT_NAME = "./FINAL set_quartergridMargined_1s/JOURNAL_GA_ism12_abs10_quarterM_1s"  # New batch results folder
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, "rooms")
        TREBLE_BASE_DIR = os.path.join(PROJECT_ROOT, "treble", "multi_experiments")
        PROJECT_DIR = os.path.join(TREBLE_BASE_DIR, PROJECT_NAME)

    # PROJECT_DIR = TREBLE_BASE_DIR  # Use TREBLE_BASE_DIR directly

    if PREPROCESS_TREBLE_RESULTS:
        if len(sys.argv) > 1:
            TREBLE_BASE_DIR = sys.argv[1]

        # First, explore an H5 file to understand its structure
        # Example: ./results/treble/single_experiments/aes_absorptioncoeffs/
        sample_dirs = [d for d in os.listdir(PROJECT_DIR)
                      if os.path.isdir(os.path.join(PROJECT_DIR, d))
                      and not d.startswith('.') and d != "processed"]

        if sample_dirs:
            # Example: ./results/treble/single_experiments/aes_absorptioncoeffs/
            sample_dir = os.path.join(PROJECT_DIR, sample_dirs[0])
            h5_files = [f for f in os.listdir(sample_dir) if f.endswith('.h5')]
            if h5_files:
                # Example: ./results/treble/single_experiments/aes_absorptioncoeffs/simulation_results.h5
                h5_file_path = os.path.join(sample_dir, h5_files[0])
                explore_h5_file(h5_file_path)

        # Process all Treble source directories with appropriate mode
        # Example input PROJECT_DIR: ./results/treble/single_experiments/
        # Example folders inside: aes_absorptioncoeffs/, aes_simtypes_scattering/, aes_sdn_treblehybrid/
        experiments_by_source, folder_names_by_source = process_all_treble_sources(PROJECT_DIR, is_singular=IS_SINGULAR)

        # Save processed data based on the mode
        if IS_SINGULAR:
            print("\nSaving experiments:")
            # Example folder_name: "aes_absorptioncoeffs" or "aes_simtypes_scattering"
            for folder_name, experiments in experiments_by_source.items():
                print(f"Saving {folder_name} with {len(experiments)} experiments")
                
                # Create a clean version of the folder name for file naming
                # Example: "aes simtypes scattering" -> "aes_simtypes_scattering"
                clean_folder_name = folder_name.replace(' ', '_')
                
                # Example output:
                # ./results/room_singulars/config_aes_absorptioncoeffs.json
                # ./results/room_singulars/rirs_aes_absorptioncoeffs.npy
                save_treble_experiments_for_singulars(
                    PROJECT_ROOT,  # Example: ./results/
                    {folder_name: experiments},
                    room_name="room_aes",
                    source_folder_name=clean_folder_name
                )
        else:
            # Example output structure for batch mode:
            # ./results/rooms/room_aes/source1/TREBLE/aes_abs20_hybrid/config.json
            # ./results/rooms/room_aes/source1/TREBLE/aes_abs20_hybrid/rirs.npy
            save_treble_experiments_matching_sdn_structure(
                PROJECT_ROOT, 
                experiments_by_source,
                PROJECT_NAME,
                room_name="journal_absorptioncoeffs"
            )

        print("\nYou can now use sdn_experiment_manager.py to visualize it.")

