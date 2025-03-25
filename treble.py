

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

def generate_receiver_grid(room_width: float, room_depth: float, n_points: int = 50):
    """Generate a grid of receiver positions within the room.

    Args:
        room_width (float): Width of the room
        room_depth (float): Depth of the room
        n_points (int): Number of receiver positions to generate

    Returns:
        List[treble.Receiver]: List of Receiver objects for the grid positions
    """
    # Create a grid of points, avoiding walls (1m margin)
    margin = 1
    x = np.linspace(margin, room_width - margin, int(np.sqrt(n_points)))
    y = np.linspace(margin, room_depth - margin, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)

    # Create a list of Receiver objects
    receivers = [
        treble.Receiver(x=pos[0], y=pos[1], z=room['mic z'], receiver_type=treble.ReceiverType.mono, label=f"mono_mic_{i}")
        for i, pos in enumerate(zip(X.flatten(), Y.flatten()))
    ]
    return receivers

def create_sources(room: dict) -> list:
    """Create a list of source positions within the room.

    Args:
        room (dict): Room parameters including dimensions and source positions.

    Returns:
        List[treble.Source]: List of Source objects for the specified positions.
    """
    sources = []
    
    # Source in the middle of the room
    sources.append(
        treble.Source(
            x=room['width'] / 2,
            y=room['depth'] / 2,
            z=room['source z'],
            source_type=treble.SourceType.omni,
            label="Center_Source",
        )
    )
    
    # Source in the lower left corner
    sources.append(
        treble.Source(
            x=1,  # 1m margin from the wall
            y=1,  # 1m margin from the wall
            z=room['source z'],
            source_type=treble.SourceType.omni,
            label="Lower_Left_Source",
        )
    )
    
    # Source in the upper right corner, offset from the right wall
    sources.append(
        treble.Source(
            x=room['width'] - 0.5,  # 0.5m offset from the right wall
            y=room['depth'] - 1,    # 1m margin from the wall
            z=room['source z'],
            source_type=treble.SourceType.omni,
            label="Upper_Right_Source",
        )
    )
    
    # Source at the top middle wall
    sources.append(
        treble.Source(
            x=room['width'] / 2,
            y=room['depth'] - 1,  # 1m margin from the top wall
            z=room['source z'],
            source_type=treble.SourceType.omni,
            label="Top_Middle_Source",
        )
    )
    
    return sources

project_name = f"SDN Treble Room Aes "
project = tsdk.get_or_create_project(name=project_name, description="SDN Treble Room Aes with flat %20 absorption but fitting abs=0.32")

# tsdk.delete_project("4ce330fd-fa65-4756-89df-2efe94019320")
my_projects = tsdk.list_my_projects()
dd.as_table(my_projects)
# dd.as_table(tsdk.geometry_library.list_datasets_with_count())


room_aes_abs02 = {'width': 9, 'depth': 7, 'height': 4,
                   'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   'absorption': 0.2,
                   }
room = room_aes_abs02


"""source_list = []
source_list.append(
    treble.Source(
        x=room['source x'],
        y=room['source y'],
        z=room['source z'],
        source_type=treble.SourceType.omni,
        label="Omni_source",
    ))
"""
source_list = create_sources(room)
receiver_list = generate_receiver_grid(room['width'], room['depth'])#room aes


# All geometry generation functions have a join_wall_layers flag which tells the generator whether walls should all share one layer
# or if each wall segment should have it's own layer.

shoebox_model = GeometryGenerator.create_shoebox_room(
    project=project,
    model_name="AES 9x7x4m shoebox join wall layers",
    width_x=room['width'],
    depth_y=room['depth'],
    height_z=room['height'],
    join_wall_layers=True,
)

shoebox_model.plot()

# absorption_coefficients = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
a = 0.2
r = (1-a)**0.5

absorption_coefficients = [a]*24
reflection_coefficients = [r]*24

# name = "SDN-AES-0.89-r_for_abs_0.2"
name = '20% Absorption'
material = tsdk.material_library.get_by_name(name)

if material is None:
    print("Material not found, creating new material")
    # First we define a material definition which is used to perform material fitting
    material_definition = treble.MaterialDefinition(
        name= name,
        description="Imported material",
        category=treble.MaterialCategory.other,
        #default_scattering=0.25,
        # material_type=treble.MaterialRequestType.third_octave_absorption,
        material_type=treble.MaterialRequestType.reflection_coefficient,
        coefficients=reflection_coefficients,
    )

    # Material fitting outputs fitted material information, nothing is saved in this step
    fitted_material = tsdk.material_library.perform_material_fitting(material_definition)
    # We can plot the material information for verification
    fitted_material.plot()

    material = tsdk.material_library.create(fitted_material)
    dd.as_tree(material)

material_assignment = [
    treble.MaterialAssignment("shoebox_walls", material),
    treble.MaterialAssignment("shoebox_floor", material),
    treble.MaterialAssignment("shoebox_ceiling", material),
]

# SDN-AES-0.2

# treble.MaterialRequestType.full_octave_absorption, \
# treble.MaterialRequestType.third_octave_absorption, \
# treble.MaterialRequestType.surface_impedance, \
# treble.MaterialRequestType.reflection_coefficient

# source_list = []
# source_list.append(
#     treble.Source(
#         x=room['source x'],
#         y=room['source y'],
#         z=room['source z'],
#         source_type=treble.SourceType.omni,
#         label="Omni_source",
#     ))

# receiver_list = []
# receiver_list.append(
#     treble.Receiver(x=room['mic x'], y=room['mic y'], z=room['mic z'], receiver_type=treble.ReceiverType.mono, label="mono_receiver")
# )

# You can use the SimulationSettings object to tune parameters of the GA solver
settings = treble.SimulationSettings(
    ga_settings=treble.GaSolverSettings(
        number_of_rays=5000,  # Number of rays to use in raytracer.
        ism_order=35,  # Image source method order.
        air_absorption=False,  # Whether to include air absorption in GA simulation.
        ism_ray_count=50000,
    )
)

"""# Additional IR length settings and GPU settings
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
)"""
# Create a simulation for each source with all receivers
simulation_definitions = []

# Create a simulation definition for each source (4 sources)
for idx, source in enumerate(source_list):
    # Use a single source with all receivers for each simulation
    sim_def = treble.SimulationDefinition(
        name=f"SDN-AES {source.label}",
        simulation_type=treble.SimulationType.hybrid,
        model=shoebox_model,  # Using your existing shoebox model
        crossover_frequency=250,
        receiver_list=receiver_list,  # All 50 receivers
        source_list=[source],         # Just this one source
        material_assignment=material_assignment,  # Your existing material assignment
        simulation_settings=settings,
        ir_length=2.0,  # 2 second IRs
    )
    
    print(f"Creating simulation for {source.label} with {len(receiver_list)} receivers")
    simulation_definitions.append(sim_def)


# Lets plot our simulation definition
#simulation_definition.plot()

#validation_results = simulation_definition.validate()
#dd.as_tree(validation_results)

#  Add a SDK simulation based on our simulation definition.
_ = project.add_simulations(simulation_definitions)
# eğer tek sim varsa : simulation = project.add_simulation(simulation_definition)
# Get cost estimate
print("Estimating simulation cost and time:")
dd.as_table(project.estimate())

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

simulations = project.get_simulations()
dd.as_table(simulations)

res = project.start_simulations()
dd.as_table(project.get_progress())

Flag_download = False

base_dir = "./results/treble/"
project_name = "room_aes_abs02"

if Flag_download:
    import os
    destination_directory = os.join(base_dir, project_name)
    project.download_results(destination_directory, rename_rule=treble.ResultRenameRule.by_label)

project = my_projects[0]
simulations = project.get_simulations()
# Get the two different simulations
simulation_0 = simulations[0]  # First simulation
print(f"Simulation 0: {simulation_0.name}")



