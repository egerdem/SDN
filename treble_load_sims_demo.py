
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


# res_obj = simulation.download_results(f'./results', rename_rule=treble.ResultRenameRule.by_label)
# res_obj.plot()

my_projects = tsdk.list_my_projects()
dd.as_table(my_projects)

project = my_projects[1]
simulations = project.get_simulations()
dd.as_table(simulations)

# Get the two different simulations
simulation_0 = simulations[0]  # First simulation
simulation_1 = simulations[1]  # Second simulation

print(f"Simulation 0: {simulation_0.name}")
print(f"Simulation 1: {simulation_1.name}")

# Define the results directory paths
res_dir1 = "/Users/ege/Projects/SDN/SDN-EGE/results"
res_dir2 = "/Users/ege/Projects/SDN/SDN-EGE/results_sim0"

# Use the correct simulation object for each results directory
results_1 = simulation_1.get_results_object(res_dir1)  # Using simulation_1 for res_dir1
results_2 = simulation_0.get_results_object(res_dir2)  # Using simulation_0 for res_dir2

# Display the results overview for each simulation
print("\nDisplaying results overview for Simulation 1:")
# results_1.plot()

print("\nDisplaying results overview for Simulation 0:")
# results_2.plot()


# results_1.plot_acoustic_parameters()
# results_2.plot_acoustic_parameters()

# Get acoustic parameters from each simulation
# params_1 = results_1.get_acoustic_parameters("Omni_source", "mono_receiver")
# params_2 = results_2.get_acoustic_parameters("Omni_source", "mono_receiver")

# Display individual parameter widgets
# print("\nDisplaying acoustic parameters for Simulation 1:")
# plot.results_parameters_plot_widget(params_1)
#
# print("\nDisplaying acoustic parameters for Simulation 0:")
# plot.results_parameters_plot_widget(params_2)

# Get the mono IRs from both simulations
treble_ir_1 = results_1.get_mono_ir("Omni_source", "mono_receiver")
treble_ir_2 = results_2.get_mono_ir("Omni_source", "mono_receiver")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')
# Get unpadded array
data = treble_ir_1.unpadded_data()
time = treble_ir_1.unpadded_time()
plt.plot(data)
plt.plot(treble_ir_1.data)
plt.legend(["Unpadded", "Padded"])
plt.show()
# Make a plot with each of those impulse responses in there
treble_ir_1.plot(comparison={"s0r0": data})

# plot.results_parameters_plot_widget(res_obj.get_acoustic_parameters(simulation.sources[0], simulation.receivers[0]))
# treble_ir = res_obj.get_mono_ir("Omni_source", "mono_receiver")
# rir_44k_treble = librosa.resample(y=treble_ir.data, orig_sr=treble_ir.sampling_rate, target_fs=44100)

# Uncomment to save the IRs if needed
# fs = 44100
# 
# # Process IR from simulation 1
# data_1 = treble_ir_1.data
# fs_treble_1 = treble_ir_1.sampling_rate
# rir_44k_treble_1 = librosa.resample(y=data_1, orig_sr=fs_treble_1, target_sr=fs)
# np.save('rir_treble_sim1.npy', rir_44k_treble_1)
# 
# # Process IR from simulation 0
# data_2 = treble_ir_2.data
# fs_treble_2 = treble_ir_2.sampling_rate
# rir_44k_treble_2 = librosa.resample(y=data_2, orig_sr=fs_treble_2, target_sr=fs)
# np.save('rir_treble_sim0.npy', rir_44k_treble_2)
