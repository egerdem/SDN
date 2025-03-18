

from treble_tsdk.tsdk import TSDK, TSDKCredentials
tsdk = TSDK(TSDKCredentials.from_file("/Users/ege/Projects/SDN/TREBLE/tsdk.cred"))

from treble_tsdk.geometry.generator import GeometryGenerator

# The tsdk_namespace provides easy access to SDK object types.
from treble_tsdk import tsdk_namespace as treble

# The display_data module provides ways to display SDK object data as trees and tables.
from treble_tsdk import display_data as dd

project_name = f"SDN Treble"
project = tsdk.get_or_create_project(name=project_name, description="SDK workshop project")

my_projects = tsdk.list_my_projects()
dd.as_table(my_projects)
# dd.as_table(tsdk.geometry_library.list_datasets_with_count())

#room aes
room = {'width': 9, 'depth': 7, 'height': 4,
                   'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   'absorption': 0.2,
                   }

# All geometry generation functions have a join_wall_layers flag which tells the generator whether walls should all share one layer
# or if each wall segment should have it's own layer.

shoebox_model = GeometryGenerator.create_shoebox_room(
    project=project,
    model_name="My 9x7x4m shoebox join wall layers",
    width_x=room['width'],
    depth_y=room['depth'],
    height_z=room['height'],
    join_wall_layers=True,
)

print(f"Layers with join_wall_layers=True: {shoebox_model.layer_names}")

shoebox_model.plot()

# absorption_coefficients = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
absorption_coefficients = [0.2]*24
# First we define a material definition which is used to perform material fitting
material_definition = treble.MaterialDefinition(
    name="SDN-AES-0.2",
    description="Imported material",
    category=treble.MaterialCategory.other,
    #default_scattering=0.25,
    material_type=treble.MaterialRequestType.third_octave_absorption,
    coefficients=absorption_coefficients,
)
# SDN-AES-0.2

# treble.MaterialRequestType.full_octave_absorption, \
# treble.MaterialRequestType.third_octave_absorption, \
# treble.MaterialRequestType.surface_impedance, \
# treble.MaterialRequestType.reflection_coefficient

# Material fitting outputs fitted material information, nothing is saved in this step
fitted_material = tsdk.material_library.perform_material_fitting(material_definition)
# We can plot the material information for verification
fitted_material.plot()

# created_material = tsdk.material_library.create(fitted_material)
# dd.as_tree(created_material)
created_material = tsdk.material_library.get_by_name("SDN-AES-0.2")


material_assignment = [
treble.MaterialAssignment("shoebox_walls", created_material),
treble.MaterialAssignment("shoebox_floor", created_material),
treble.MaterialAssignment("shoebox_ceiling", created_material),
]
source_list = []
source_list.append(
    treble.Source(
        x=room['source x'],
        y=room['source y'],
        z=room['source z'],
        source_type=treble.SourceType.omni,
        label="Omni_source",
    ))

receiver_list = []
receiver_list.append(
    treble.Receiver(x=room['mic x'], y=room['mic y'], z=room['mic z'], receiver_type=treble.ReceiverType.mono, label="mono_receiver")
)

# You can use the SimulationSettings object to tune parameters of the GA solver
settings = treble.SimulationSettings(
    ga_settings=treble.GaSolverSettings(
        number_of_rays=5000,  # Number of rays to use in raytracer.
        ism_order=12,  # Image source method order.
        air_absorption=False,  # Whether to include air absorption in GA simulation.
        ism_ray_count=50000,
    )
)

# Additional IR length settings and GPU settings
simulation_definition = treble.SimulationDefinition(
    name="SDN-AES simulation ism order 12",
    simulation_type=treble.SimulationType.hybrid,
    model=shoebox_model,
    crossover_frequency=250,
    # energy_decay_threshold=20,  # Instead of a constant IR length you can define a energy decay threshold which will stop the simulation automatically.
    receiver_list=receiver_list,
    source_list=source_list,
    material_assignment=material_assignment,
    simulation_settings=settings,
    ir_length= 2.0,
)

# Lets plot our simulation definition
simulation_definition.plot()

validation_results = simulation_definition.validate()
dd.as_tree(validation_results)

#  Add a SDK simulation based on our simulation definition.
simulation = project.add_simulation(simulation_definition)

# Get an estimate of simulation runtime and cost.
# Note: estimate is not available until a mesh has been generated from the model. The wait_for_estimate
#       methods waits until the mesh is ready before getting the estimate.
# simulation_estimation = simulation.wait_for_estimate()
# simulation_estimation.as_tree()

# Lets start the simulation, the SDK will then allocate cloud GPUs and CPUs resources to your simulation.
# simulation.start()

# It's possible to view the progress of the simulation by calling either .get_progress() or viewing it as a 'live' view
# using .as_live_progress().
# simulation.as_live_progress()


# _ = project.add_simulations(simulation_definition)

dd.as_table(project.estimate())

# res = project.start_simulations()
# dd.as_table(project.get_progress())

simulations = project.get_simulations()
dd.as_table(simulations)

res_obj = simulation.download_results(f'./results', rename_rule=treble.ResultRenameRule.by_label)
res_obj.plot()

from treble_tsdk.results import plot
plot.results_parameters_plot_widget(res_obj.get_acoustic_parameters(simulation.sources[0], simulation.receivers[0]))

treble_ir = res_obj.get_mono_ir("Omni_source", "mono_receiver")

import librosa
fs = 44100
data = treble_ir.data
fs_treble = treble_ir.sampling_rate

treble_ir = res_obj.get_mono_ir("Omni_source", "mono_receiver")

rir_44k_treble = librosa.resample(y=data, orig_sr=fs_treble, target_fs=fs)


