

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

def generate_receiver_grid(room_width: float, room_depth: float, margin = 1, n_points: int = 50):
    """Generate a grid of receiver positions within the room.

    Args:
        room_width (float): Width of the room
        room_depth (float): Depth of the room
        n_points (int): Number of receiver positions to generate

    Returns:
        List[treble.Receiver]: List of Receiver objects for the grid positions
    """
    # Create a grid of points, avoiding walls (1m margin)
    # margin = 1
    margin_from_center = margin - 0.2
    x = np.linspace(margin, room_width - margin_from_center, int(np.sqrt(n_points)))
    y = np.linspace(margin, room_depth - margin_from_center, int(np.sqrt(n_points)))
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

project_name = f"FINAL set_quartergridMargined_SCAT06"
project = tsdk.get_or_create_project(name=project_name, description="original location, single")

# tsdk.delete_project("dcdc0b0b-3c2e-4441-8c9b-3888de56279c")
my_projects = tsdk.list_my_projects()
dd.as_table(my_projects)

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

room_waspaa = {
        'width': 6, 'depth': 4, 'height': 7,
        'source x': 3.6, 'source y': 1.3, 'source z': 5.3,
        'mic x': 1.2, 'mic y': 2.4, 'mic z': 1.8,
        'absorption': 0.1,
    }

room = room_journal

# Multiple Sources and Receivers
source_list = create_sources(room)
# receiver_list = generate_receiver_grid(room['width'], room['depth'], n_points=50) #room aes

# retrieve only the left bottom quarter of the room
receiver_list = generate_receiver_grid(room['width']/2, room['depth']/2, n_points=16, margin=0.5) #room aes

# plot the receivers in 2d using matplotlib
"""import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
receiver_positions = [(r.x, r.y) for r in receiver_list]
receiver_positions = np.array(receiver_positions)
plt.scatter(receiver_positions[:, 0], receiver_positions[:, 1])
#also plot the room as a rectangle, no z.
plt.gca().add_patch(plt.Rectangle((0, 0), room['width'], room['depth'], fill=None))
plt.show()"""

# All geometry generation functions have a join_wall_layers flag which tells the generator whether walls should all share one layer
# or if each wall segment should have it's own layer.

shoebox_model = GeometryGenerator.create_shoebox_room(
    project=project,
    model_name="shoebox join wall layers",
    width_x=room['width'],
    depth_y=room['depth'],
    height_z=room['height'],
    join_wall_layers=True,
)

shoebox_model.plot()

# absorption_coefficients = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
# a = 0.3
# r = (1-a)**0.5

# absorption_coefficients = [a]*8
# reflection_coefficients = [r]*24

# name = "SDN-AES-0.89-r_for_abs_0.2"
# name = '20% Absorption'
# name = '20% Absorption with 1 Scattering'
name = '20% Absorption with 1 Scattering'
# name = "Reflection of 10% Absorption"
material = tsdk.material_library.get_by_name(name)
z = tsdk.material_library.get() # get all materials

"""if material is None:
    print("Material not found, creating new material")
    # First we define a material definition which is used to perform material fitting
    material_definition = treble.MaterialDefinition(
        name= name,
        description="Imported material",
        category=treble.MaterialCategory.other,
        default_scattering=1,
        # material_type=treble.MaterialRequestType.third_octave_absorption,
        material_type=treble.MaterialRequestType.full_octave_absorption,
        coefficients=absorption_coefficients,
    )
#
#     # Material fitting outputs fitted material information, nothing is saved in this step
    fitted_material = tsdk.material_library.perform_material_fitting(material_definition)
#     # We can plot the material information for verification
#     fitted_material.plot()
#
    material = tsdk.material_library.create(fitted_material)
    dd.as_tree(material)

else:
    print("Material found")
    dd.as_tree(material)"""

material_assignment = [
    treble.MaterialAssignment("shoebox_walls", material),
    treble.MaterialAssignment("shoebox_floor", material),
    treble.MaterialAssignment("shoebox_ceiling", material),
]

# treble.MaterialRequestType.full_octave_absorption, \
# treble.MaterialRequestType.third_octave_absorption, \
# treble.MaterialRequestType.surface_impedance, \
# treble.MaterialRequestType.reflection_coefficient

# You can use the SimulationSettings object to tune parameters of the GA solver
settings = treble.SimulationSettings(
    ga_settings=treble.GaSolverSettings(
        number_of_rays=5000,  # Number of rays to use in raytracer.
        ism_order=12,  # Image source method order.
        air_absorption=False,  # Whether to include air absorption in GA simulation.
        ism_ray_count=50000,
    )
)

# Create a simulation for each source with all receivers
simulation_definitions = []

# MULTI SIM Create a simulation definition for each source (4 sources)
for idx, source in enumerate(source_list):
    # Use a single source with all receivers for each simulation
    sim_def = treble.SimulationDefinition(
        name=f"JOURNAL_GA_ism12_abs20_scat1_quarterM_1s_{source.label}",
        simulation_type=treble.SimulationType.ga,
        model=shoebox_model,  # Using your existing shoebox model
        # crossover_frequency=250,
        receiver_list=receiver_list,  # All ** receivers
        source_list=[source],         # Just this one source
        material_assignment=material_assignment,  # Your existing material assignment
        simulation_settings=settings,
        ir_length=1.0,  # 2 second IRs
    )
    
    print(f"Creating simulation for {source.label} with {len(receiver_list)} receivers")
    simulation_definitions.append(sim_def)

_ = project.add_simulations(simulation_definitions)
simulations = project.get_simulations()
dd.as_table(simulations)

"""# Create a simulation for each source with all receivers
simulation_definitions = []

# MULTI SIM Create a simulation definition for each source (4 sources)
for idx, source in enumerate(source_list):
    # Use a single source with all receivers for each simulation
    sim_def = treble.SimulationDefinition(
        name=f"AES_GA_ism12_abs20_quarterM_1s_{source.label}",
        simulation_type=treble.SimulationType.ga,
        model=shoebox_model,  # Using your existing shoebox model
        crossover_frequency=250,
        receiver_list=receiver_list,  # All ** receivers
        source_list=[source],  # Just this one source
        material_assignment=material_assignment,  # Your existing material assignment
        simulation_settings=settings,
        ir_length=1.0,  # 2 second IRs
    )

    print(f"Creating simulation for {source.label} with {len(receiver_list)} receivers")
    simulation_definitions.append(sim_def)

_ = project.add_simulations(simulation_definitions)

# Create a simulation for each source with all receivers
simulation_definitions = []

# MULTI SIM Create a simulation definition for each source (4 sources)
for idx, source in enumerate(source_list):
    # Use a single source with all receivers for each simulation
    sim_def = treble.SimulationDefinition(
        name=f"AES_DG_Reflect_abs20_multi_1sec_{source.label}",
        simulation_type=treble.SimulationType.dg,
        model=shoebox_model,  # Using your existing shoebox model
        crossover_frequency=250,
        receiver_list=receiver_list,  # All ** receivers
        source_list=[source],  # Just this one source
        material_assignment=material_assignment,  # Your existing material assignment
        simulation_settings=settings,
        ir_length=1.0,  # 2 second IRs
    )

    print(f"Creating simulation for {source.label} with {len(receiver_list)} receivers")
    simulation_definitions.append(sim_def)"""


#  MULTIPLE SIMS: Add a SDK simulation based on our simulation definition.
# for sim in simulation_definitions:
#     simulation = project.add_simulation(sim)
    # simulation.start()
    # simulation.as_live_progress()

# simulations = project.get_simulations()
# dd.as_table(simulations)

res = project.start_simulations()
# dd.as_table(project.get_progress())
dd.as_table(project.as_live_progress())

Flag_download = True

base_dir = "./results/treble/multi_experiments/"
# base_dir = "./results/treble/final_set_qM_scat06_1s/"
# base_dir = "./results/treble/final_set_qM_scat1_1s/"

my_projects = tsdk.list_my_projects()
project = my_projects[0]

if Flag_download:

    import os
    destination_directory = os.path.join(base_dir, project_name)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    project.download_results(destination_directory, rename_rule=treble.ResultRenameRule.by_label)

    # for sim in simu:
    #     project_name = sim.name
    #     destination_directory = os.path.join(base_dir, project_name)
    #     # project.download_results(destination_directory, rename_rule=treble.ResultRenameRule.by_label)
    #     sim.download_results(destination_directory, rename_rule=treble.ResultRenameRule.by_label)





