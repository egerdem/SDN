
# ./results/
#     ├── treble/
#     │   ├── single_experiments/          # Raw Treble results
#     │   └── multi_experiments/           # Raw Treble batch results
#     │
#     └── room_singulars/                  # All single experiments
#         ├── PREPROCESSED/                # Temporary location for processed Treble results
#         ├── aes_absorptioncoeffs/        # SDN/ISM experiments group 1
#         ├── aes_simtypes_scattering/     # SDN/ISM experiments group 2
#         ├── aes_sdn_treblehybrid/        # SDN/ISM experiments group 3
#         └── room_info_aes.json           # Room configuration

from treble_tsdk.tsdk import TSDK, TSDKCredentials
tsdk = TSDK(TSDKCredentials.from_file("/Users/ege/Projects/SDN/TREBLE/tsdk.cred"))
from treble_tsdk.geometry.generator import GeometryGenerator
# The tsdk_namespace provides easy access to SDK object types.
from treble_tsdk import tsdk_namespace as treble
# The display_data module provides ways to display SDK object data as trees and tables.
from treble_tsdk import display_data as dd
from treble_tsdk.results import plot
import librosa
import numpy as np

project_name = f"AES_single_scattering-ga-dg-hybrid"
project = tsdk.get_or_create_project(name=project_name, description="original location, single")

room_waspaa = {
        'width': 6, 'depth': 4, 'height': 7,
        'source x': 3.6, 'source y': 1.3, 'source z': 5.3,
        'mic x': 1.2, 'mic y': 2.4, 'mic z': 1.8,
        'absorption': 0.1,
    }

room_aes = {'width': 9, 'depth': 7, 'height': 4,
                   'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   'absorption': 0.2,
                   }

room_journal = {'width': 3.2, 'depth': 4, 'height': 2.7,
                   'source x': 2, 'source y': 3., 'source z': 2,
                   'mic x': 1, 'mic y': 1, 'mic z': 1.5,
                   'absorption': 0.1,
                   }

room = room_aes

a = room['absorption']
r = (1-a)**0.5

absorption_coefficients = [a]*8
reflection_coefficients = [r]*8

name = '20% Absorption'
# name = "Reflection of 30% Absorption"
# name = '20% Absorption with 1 Scattering'
# name = '20% Absorption with 0.6 Scattering'
material = tsdk.material_library.get_by_name(name)
# z = tsdk.material_library.get() # get all materials

# if material is None:
#     print("Material not found, creating new material")
#     # First we define a material definition which is used to perform material fitting
#     material_definition = treble.MaterialDefinition(
#         name= name,
#         description="Imported material",
#         category=treble.MaterialCategory.other,
#         default_scattering=1,
#         # material_type=treble.MaterialRequestType.third_octave_absorption,
#         material_type=treble.MaterialRequestType.full_octave_absorption,
#         coefficients=absorption_coefficients,
#     )
# #
# #     # Material fitting outputs fitted material information, nothing is saved in this step
#     fitted_material = tsdk.material_library.perform_material_fitting(material_definition)
# #     # We can plot the material information for verification
#     fitted_material.plot()
# #
#     material = tsdk.material_library.create(fitted_material)
#     dd.as_tree(material)

material_assignment = [
    treble.MaterialAssignment("shoebox_walls", material),
    treble.MaterialAssignment("shoebox_floor", material),
    treble.MaterialAssignment("shoebox_ceiling", material),
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
    treble.Receiver(x=room['mic x'], y=room['mic y'], z=room['mic z'], receiver_type=treble.ReceiverType.mono, label="mono_receiver"))

shoebox_model = GeometryGenerator.create_shoebox_room(
    project=project,
    model_name="aes",
    width_x=room['width'],
    depth_y=room['depth'],
    height_z=room['height'],
    join_wall_layers=True,
)

settings = treble.SimulationSettings(
    ga_settings=treble.GaSolverSettings(
        number_of_rays=5000,  # Number of rays to use in raytracer.
        ism_order=12,  # Image source method order.
        air_absorption=False,  # Whether to include air absorption in GA simulation.
        ism_ray_count=50000,
    )
)

simulation_definitions = []

simulation_definition = treble.SimulationDefinition(
    name="aes_single_ga_abs20_ism12_1sec",
    simulation_type=treble.SimulationType.ga,
    model=shoebox_model,
    crossover_frequency=250,
    # energy_decay_threshold=20,  # Instead of a constant IR length you can define a energy decay threshold which will stop the simulation automatically.
    receiver_list=receiver_list,
    source_list=source_list,
    material_assignment=material_assignment,
    simulation_settings=settings,
    ir_length= 1.0)

simulation_definitions.append(simulation_definition)

simulation_definition = treble.SimulationDefinition(
    name="aes_single_hybrid_abs20_ism12_1sec",
    simulation_type=treble.SimulationType.hybrid,
    model=shoebox_model,
    crossover_frequency=250,
    # energy_decay_threshold=20,  # Instead of a constant IR length you can define a energy decay threshold which will stop the simulation automatically.
    receiver_list=receiver_list,
    source_list=source_list,
    material_assignment=material_assignment,
    simulation_settings=settings,
    ir_length= 1.0)

simulation_definitions.append(simulation_definition)

# simulation = project.add_simulation(simulation_definition)
# dd.as_table(project.estimate())

# Get an estimate of simulation runtime and cost.
# Note: estimate is not available until a mesh has been generated from the model. The wait_for_estimate
#       methods waits until the mesh is ready before getting the estimate.
# simulation_estimation = simulation.wait_for_estimate()
# simulation_estimation.as_tree()

#SINGLE
# Lets start the simulation, the SDK will then allocate cloud GPUs and CPUs resources to your simulation.
# simulation.start()
# simulation.as_live_progress()

# _ = project.add_simulations(simulation_definitions)

Start_simulations = True

if Start_simulations:
    for sim in simulation_definitions:
        simulation = project.add_simulation(sim)
        # simulation.start()
        # simulation.as_live_progress()

simulations = project.get_simulations()
dd.as_table(simulations)

#start project sims
project.start_simulations()
dd.as_table(project.get_progress())

Flag_download = True

base_dir = "./results/treble/"
project_name = "single_experiments"

if Flag_download:
    import os

    destination_directory = os.path.join(base_dir, project_name, project.name)
    #if not exist, create destination directory
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    project.download_results(destination_directory, rename_rule=treble.ResultRenameRule.by_label)

