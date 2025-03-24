
import librosa
import numpy as np

from treble_tsdk.tsdk import TSDK, TSDKCredentials
tsdk = TSDK(TSDKCredentials.from_file("/Users/ege/Projects/SDN/TREBLE/tsdk.cred"))
from treble_tsdk.geometry.generator import GeometryGenerator
# The tsdk_namespace provides easy access to SDK object types.
from treble_tsdk import tsdk_namespace as treble
# The display_data module provides ways to display SDK object data as trees and tables.
from treble_tsdk import display_data as dd
from treble_tsdk.results import plot
import json

my_projects = tsdk.list_my_projects()
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
    source_positions[source["label"]] = source_position

#also log "name", "simulationType",  "layerMaterialAssignments"[0]."materialName"


# Define the results directory paths
# results_0 = simulation_0.get_results_object(source_dir)
# results_0.plot()
# treble_ir_1 = results_0.get_mono_ir("Omni_source", "mono_receiver")


