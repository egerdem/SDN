
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

my_projects = tsdk.list_my_projects()
# dd.as_table(my_projects)

project = my_projects[0]
simulations = project.get_simulations()
# destination_directory = "./results/treble/"
# project.download_results(destination_directory, rename_rule=treble.ResultRenameRule.by_label)

# Get the two different simulations
simulation_0 = simulations[0]  # First simulation
simulation_1 = simulations[1]  # Second simulation

print(f"Simulation 0: {simulation_0.name}")

# Define the results directory paths
results_1 = simulation_0.get_results_object("./results/treble/SDN-AES Center_Source")  # Using simulation_1 for res_dir1
results_1.plot()