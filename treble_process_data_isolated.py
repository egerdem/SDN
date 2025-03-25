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
        'absorption': 0.2  # Extract from material properties if possible
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

    # Extract simulation parameters
    duration = simulation_info.get("impulseLengthSec")
    sim_type = simulation_info.get("simulationType", "empty")
    sim_settings = simulation_info.get("simulationSettings", {})
    crossover = simulation_info.get("crossoverFrequency", 250)

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
        tuple: (source_label, list of TrebleExperiment objects)
    """
    # Find the simulation_info.json file
    sim_info_path = os.path.join(source_dir, "simulation_info.json")
    if not os.path.exists(sim_info_path):
        print(f"No simulation_info.json found in {source_dir}")
        return None, []

    # Load simulation info
    with open(sim_info_path, 'r') as f:
        simulation_info = json.load(f)

    # Find the H5 file
    h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]
    if not h5_files:
        print(f"No H5 file found in {source_dir}")
        return None, []

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

    return source_label, experiments


def process_all_treble_sources(project_dir):
    """
    Process all Treble results in the base directory.

    Args:
        treble_base_dir: Path to the base directory containing Treble results

    Returns:
        dict: Mapping from source label to list of TrebleExperiment objects
    """
    if not os.path.exists(project_dir):
        print(f"Treble base directory {project_dir} not found")
        return {}

    experiments_by_source = {}

    # Process each source directory
    for item in os.listdir(project_dir):
        source_dir = os.path.join(project_dir, item)
        if os.path.isdir(source_dir) and not item.startswith('.') and item != "processed":
            print(f"Processing Treble source directory: {item}")
            source_label, experiments = process_treble_source_directory(source_dir)
            if source_label and experiments:
                experiments_by_source[source_label] = experiments

    return experiments_by_source


def save_treble_experiments_matching_sdn_structure(treble_base_dir, experiments_by_source, room_name):
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

# Main function
if __name__ == "__main__":
    import sys
    import os
    global_results_dir = "./results"
    # Process command line arguments
    treble_base_dir = "./results/treble"
    project_name = "room_aes_abs02" # this is the original treble results folder
    project_dir = os.path.join(treble_base_dir, project_name)

    preprocess_treble_results = True
    # load_preprocessed_data_to_sdn_batch_manager = False

    if preprocess_treble_results:

        if len(sys.argv) > 1:
            treble_base_dir = sys.argv[1]

        # First, explore an H5 file to understand its structure
        sample_dirs = [d for d in os.listdir(project_dir)
                       if os.path.isdir(os.path.join(treble_base_dir, d))
                       and not d.startswith('.') and d != "processed"]

        if sample_dirs:
            sample_dir = os.path.join(project_dir, sample_dirs[0])
            h5_files = [f for f in os.listdir(sample_dir) if f.endswith('.h5')]
            if h5_files:
                h5_file_path = os.path.join(sample_dir, h5_files[0])
                explore_h5_file(h5_file_path)

        # Process all Treble source directories
        experiments_by_source = process_all_treble_sources(project_dir)

        # Save processed data for later loading by SDN batch manager
        # save_treble_experiments(project_dir, experiments_by_source) #old method
        save_treble_experiments_matching_sdn_structure(global_results_dir, experiments_by_source, room_name = "room_aes")

        # Print summary
        total_exps = sum(len(exps) for exps in experiments_by_source.values())
        print(f"\nTotal experiments processed: {total_exps}")
        for source_label, experiments in experiments_by_source.items():
            print(f"Source {source_label}: {len(experiments)} experiments")

        print("\nTreble data has been processed and saved. You can now use sdn_experiment_manager.py to visualize it.")

