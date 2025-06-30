from collections import deque
import numpy as np
from typing import Dict, List, Optional
from geometry import Room, get_best_node2node_targets, get_best_reflection_targets, build_specular_matrices_from_angles, get_image_sources
from path_logger import PathLogger, PressurePacket, deque_plotter
import matplotlib.pyplot as plt
import specular_scattering_matrix as ssm
import random

def random_wall_mapping(walls):
    """
    Given a list of unique wall‐IDs, return a dict that maps each wall
    to a different one, with no wall mapped to itself.

    >>> walls = ["e", "w", "s", "f", "n", "c"]
    >>> random_wall_mapping(walls)
    {'e': 'n', 'w': 'c', 's': 'e', 'f': 'w', 'n': 's', 'c': 'f'}
    """
    if len(set(walls)) != len(walls):
        raise ValueError("Wall IDs must be unique")

    while True:
        shuffled = walls[:]        # copy
        random.shuffle(shuffled)   # in-place random permutation

        # keep the permutation only if nothing stayed in place
        if all(orig != new for orig, new in zip(walls, shuffled)):
            return dict(zip(walls, shuffled))

def scattering_matrix_crate(increase_coef=0.2):
    """Create a modified scattering matrix with adjusted diagonal and off-diagonal elements.
    
    Args:
        increase_coef (float): Coefficient to adjust the matrix elements
        
    Returns:
        np.ndarray: Modified scattering matrix
    """
    c = increase_coef
    original_matrix = (2 / 5) * np.ones((5, 5)) - np.eye(5)
    adjusted_matrix = np.copy(original_matrix)
    
    # Decrease diagonal elements by c
    np.fill_diagonal(adjusted_matrix, adjusted_matrix.diagonal() - c)
    
    # Increase only off-diagonal elements by c/4
    off_diagonal_mask = np.ones(adjusted_matrix.shape, dtype=bool)
    np.fill_diagonal(off_diagonal_mask, False)
    
    # Add c/4 to only the off-diagonal elements
    adjusted_matrix[off_diagonal_mask] += (c / 4)
    return adjusted_matrix


def swap(s, j, k):
    """Swap two rows in a matrix."""
    s_jk = s.copy()
    s_jk[[j, k]] = s_jk[[k, j]]
    return s_jk

class DelayNetwork:
    """Core SDN implementation focusing on sample-based processing with accessible delay lines."""

    def __init__(self, room: Room, Fs: int = 44100, c: float = 343.0, source_pressure_injection_coeff: float = 0.5,
                 coef: float = 2.0/5,
                 source_weighting: float = 1,
                 injection_c_vector: List[float] = None,
                 injection_vector: List[float] = None,
                 use_identity_scattering: bool = False,
                 specular_scattering: bool = False,
                 specular_increase_coef: float = 0.0,
                 specular_source_injection: bool = False,
                 specular_source_injection_random: bool = False,
                 specular_node_pressure: bool = False,
                 ignore_wall_absorption: bool = False,
                 ignore_src_node_atten: bool = False,
                 ignore_node_mic_atten: bool = False,
                 enable_path_logging: bool = False,
                 scattering_matrix_update_coef: float = None,
                 source_trial_injection: Optional[float] = None,
                 more_absorption: bool = False,
                 print_mic_pressures: bool = False,
                 print_parameter_summary: bool = False,
                 print_weighted_psk: bool = False,
                 normalize_to_first_impulse: bool = False,
                 ho_sdn_order: Optional[int] = None,
                 label: str = ""):
        """Initialize SDN with test flags.
        
        Args:
            room: Room geometry and parameters
            Fs: Sampling frequency (default: 44100)
            c: Speed of sound (default: 343.0)
            use_identity_scattering: If True, use identity matrix for scattering (default: False)
            specular_scattering: If True, use specular reflection matrix (default: False)
            specular_source_injection: If True, inject source pressure to random node (default: False)
            ignore_wall_absorption: If True, set wall reflection coefficients to 1 (default: False)
            ignore_src_node_atten: If True, set source-to-node gains to 1 (default: False)
            ignore_node_mic_atten: If True, set node-to-mic gains to 1 (default: False)
            scattering_matrix_update_coef: If not None, use modified scattering matrix with this coefficient
            specular_increase_coef: Coefficient for build_specular_matrices_from_angles if specular_scattering is True.
            ho_sdn_order: If not None, specifies the order for HO-SDN, activating the model (default: None).
        """
        self.room = room
        self.Fs = Fs
        self.c = c
        self.source_pressure_injection_coeff = source_pressure_injection_coeff
        self.coef = coef
        self.source_weighting = source_weighting
        self.injection_c_vector = injection_c_vector
        self.injection_vector = injection_vector
        # Test flags
        self.use_identity_scattering = use_identity_scattering
        self.specular_scattering = specular_scattering
        self.specular_source_injection = specular_source_injection
        self.specular_source_injection_random = specular_source_injection_random
        self.specular_node_pressure = specular_node_pressure
        self.ignore_wall_absorption = ignore_wall_absorption
        self.ignore_src_node_atten = ignore_src_node_atten
        self.ignore_node_mic_atten = ignore_node_mic_atten
        self.more_absorption = more_absorption
        self.scattering_matrix_update_coef = scattering_matrix_update_coef
        self.print_mic_pressures = print_mic_pressures
        self.print_parameter_summary = print_parameter_summary
        self.print_weighted_psk = print_weighted_psk
        self.specular_increase_coef = specular_increase_coef
        self.ho_sdn_order = ho_sdn_order
        self.source_trial_injection = source_trial_injection
        # Override wall attenuation if ignore_wall_absorption is True
        if self.ignore_wall_absorption:
            self.room.wallAttenuation = [1.0] * len(self.room.wallAttenuation)

        self.label = label
        self.injection_index = 0
        self.total_injection_count = 6
        self.non_dominant_index = 1  # Track which index to use for non-dominant nodes

        if self.specular_source_injection_random:
            self.random_target_map = random_wall_mapping(list(self.room.walls.keys()))
            print("Random wall mapping created:")
            print(self.random_target_map)

        # Print parameter summary
        if self.print_parameter_summary:
            self._print_parameter_summary()

        # SDN parameters
        self.num_nodes = len(room.walls)  # Number of scattering nodes (6 for cuboid)

        # Initialize scattering matrices
        self.scattering_matrix = self._create_scattering_matrix()

        # Initialize delay lines with public access using descriptive keys
        self.source_to_mic = {}  # Direct source to microphone
        self.source_to_nodes = {}  # Format: "src_to_{wall_id}"
        self.node_to_mic = {}  # Format: "{wall_id}_to_mic"
        self.node_to_node = {}  # Format: "{wall1_id}_to_{wall2_id}"

        if self.ho_sdn_order is not None:
            print(f"--- Initializing HO-SDN of order {self.ho_sdn_order} ---")
            all_image_sources = get_image_sources(self.room, self.ho_sdn_order)
            print("all_image_sources:", len(all_image_sources))
            # Early reflections (order < N) are handled as direct paths to the mic
            self.early_reflection_paths = [img for img in all_image_sources if img['order'] < self.ho_sdn_order and img['order'] > 0]
            self.early_reflection_lines = {} # "path_idx" -> deque
            
            # Order-N reflections feed the SDN network
            self.sdn_feeding_paths = [img for img in all_image_sources if img['order'] == self.ho_sdn_order]
            self.ho_source_to_nodes = {}  # Format: "src_img_{img_idx}_to_{wall_id}"
        else:
            self.early_reflection_paths = []
            self.sdn_feeding_paths = []
            self.ho_source_to_nodes = None

        self._setup_delay_lines()

        # Initialize attenuation factors with matching keys
        self.source_to_node_gains = {}  # g_sk, Format: "src_to_{wall_id}"
        self.node_to_mic_gains = {}  # g_km, Format: "{wall_id}_to_mic"
        self._calculate_attenuations()

        # State variables
        self.node_pressures = {wall_id: 0.0 for wall_id in room.walls}

        # For storing outgoing wave variables at each node
        self.outgoing_waves = {wall_id: {} for wall_id in room.walls}
        for wall1_id in room.walls:
            for wall2_id in room.walls:
                if wall1_id != wall2_id:
                    self.outgoing_waves[wall1_id][wall2_id] = 0.0

        self.wall_incoming_sums = {wall_id: [] for wall_id in room.walls}
        self.sample_indices = {wall_id: [] for wall_id in room.walls}  # To store n values for plotting


    def _print_parameter_summary(self):
        """Print a formatted summary of only the non-default SDN parameters."""
        print("\n" + "=" * 50)
        print(f"SDN Configuration Summary for: {self.label}")

        # Define default values
        defaults = {
            'source_pressure_injection_coeff': 0.5,
            'coef': 2.0 / 5,
            'source_weighting': 1,
            'use_identity_scattering': False,
            'specular_scattering': False,
            'specular_source_injection': False,
            'ignore_wall_absorption': False,
            'ignore_src_node_atten': False,
            'ignore_node_mic_atten': False,
            'enable_path_logging': False,
            'scattering_matrix_update_coef': None,
        }

        # Check and print only non-default values
        non_default_params = []

        if self.source_pressure_injection_coeff != defaults['source_pressure_injection_coeff']:
            non_default_params.append(
                f"• Source Pressure Coefficient (1/2 originally): {self.source_pressure_injection_coeff}")

        if self.coef != defaults['coef']:
            non_default_params.append(f"• Coefficient in front of Summation Symbol (2/5 originally): {self.coef}")

        if self.source_weighting != defaults['source_weighting']:
            non_default_params.append(f"• Source Weighting: {self.source_weighting}")

        # Scattering Configuration
        if self.use_identity_scattering:
            non_default_params.append("• Using Identity Scattering Matrix")
        elif self.specular_scattering:
            non_default_params.append("• Using Specular Scattering Matrix")
        elif self.scattering_matrix_update_coef is not None:
            non_default_params.append(
                f"• Using Modified Scattering Matrix (coef: {self.scattering_matrix_update_coef})")

        # Source Injection Mode
        if self.specular_source_injection:
            if self.specular_source_injection_random:
                non_default_params.append("• Using Random Specular Source Injection")
            else:
                non_default_params.append("• Using Specular Source Injection")
            if self.source_weighting != defaults['source_weighting']:
                non_default_params.append(f"  - Source Weighting Factor: {self.source_weighting}")
            
            # Print best and second-best targets for each wall
            non_default_params.append("  - Best and Second-Best Reflection Targets:")
            for wall_id in self.room.walls:
                targets = get_best_reflection_targets(wall_id, self.room.angle_mappings, num_targets=2)
                if targets:
                    best_target = targets[0]
                    second_best = targets[1] if len(targets) > 1 else "None"
                    non_default_params.append(f"    - {wall_id} -> {best_target} (best), {second_best} (second-best)")

        # Attenuation Settings
        if self.ignore_wall_absorption:
            non_default_params.append("• Wall Absorption Ignored")
        if self.ignore_src_node_atten:
            non_default_params.append("• Source-Node Attenuation Ignored")
        if self.ignore_node_mic_atten:
            non_default_params.append("• Node-Mic Attenuation Ignored")

        # Print all non-default parameters
        if non_default_params:
            print("\nParameters:")
            print("\n".join(non_default_params))
        else:
            print("\nAll parameters are at default values")

        print("=" * 50 + "\n")

    def _create_scattering_matrix(self) -> np.ndarray:
        """Create the initial scattering matrix based on current settings.
        
        The matrix is either:
        - Identity matrix if use_identity_scattering is True
        - Specular matrix if specular_scattering is True
        - Modified matrix if scattering_matrix_update_coef is not None
        - Diffuse matrix (default) otherwise
        
        Returns:
            np.ndarray: The appropriate scattering matrix
        """
        N = self.num_nodes
        size = N - 1  # Matrix size for each node's operation

        if self.use_identity_scattering:
            return np.eye(size)
        elif self.specular_scattering:
            print(f"Specular scattering matrix with fixed angles,increase_coef:{self.specular_increase_coef}\n")
            return build_specular_matrices_from_angles(self.room, increase_coef=self.specular_increase_coef)
        elif self.scattering_matrix_update_coef is not None:
            print(scattering_matrix_crate(self.scattering_matrix_update_coef))
            return scattering_matrix_crate(self.scattering_matrix_update_coef)
        else:
            # print("Default diffuse scattering matrix is used\n")
            # Default diffuse, original scattering
            return (2 / size) * np.ones((size, size)) - np.eye(size)

    def _create_specular_scattering_matrices(self) -> Dict[str, np.ndarray]:
        """Create specular scattering matrices for all walls.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping wall IDs to their specular matrices
        """
        return build_specular_matrices_from_angles(self.room)

    def _setup_delay_lines(self):
        """Initialize all delay lines in the network."""

        # Direct source to microphone
        key = "src_to_mic"
        src_mic_distance = self.room.source.srcPos.getDistance(self.room.micPos)
        self.direct_sound_delay = int(np.ceil(((self.Fs * src_mic_distance) / self.c)))

        self.source_to_mic[key] = deque([0.0] * self.direct_sound_delay, maxlen=self.direct_sound_delay)

        # Setup based on whether HO-SDN is active
        if self.ho_sdn_order is not None:
            # HO-SDN Case
            # 1. Setup early reflection paths (order < N) as direct source-mic lines
            for i, path_info in enumerate(self.early_reflection_paths):
                dist = path_info['position'].getDistance(self.room.micPos)
                delay = round((self.Fs * dist) / self.c)
                key = f"early_path_{i}"
                self.early_reflection_lines[key] = deque([0.0] * delay, maxlen=delay)

            # 2. Setup Order-N reflection paths to feed the SDN nodes
            for i, img_info in enumerate(self.sdn_feeding_paths):
                last_wall_id = img_info['path'][-1]
                wall = self.room.walls[last_wall_id]
                
                # Correct delay calculation for the source-to-node path (alpha_k)
                # alpha_k = delta_true - beta_k
                delta_true = img_info['position'].getDistance(self.room.micPos)
                beta_k = wall.node_positions.getDistance(self.room.micPos)
                alpha_k_dist = delta_true - beta_k

                # Ensure delay is not negative due to geometric/numerical precision issues
                if alpha_k_dist < 0:
                    alpha_k_dist = 0
                
                delay = round((self.Fs * alpha_k_dist) / self.c)
                key = f"sdn_feed_path_{i}_to_{last_wall_id}"
                self.ho_source_to_nodes[key] = deque([0.0] * delay, maxlen=delay)
                setattr(self, f'{key}_delay', delay)

        else:
            # Standard SDN Case: Source to nodes (1st order)
            for wall_id, wall in self.room.walls.items():
                key = f"src_to_{wall_id}"
                dist = wall.node_positions.getDistance(self.room.source.srcPos)
                delay_src2node = round((self.Fs * dist) / self.c)
                setattr(self, f'src_to_{wall_id}_delay', delay_src2node)

                self.source_to_nodes[key] = deque([0.0] * delay_src2node, maxlen=delay_src2node)

        # Nodes to microphone (common for both standard and HO-SDN)
        for wall_id, wall in self.room.walls.items():
            key = f"{wall_id}_to_mic"
            dist = wall.node_positions.getDistance(self.room.micPos)
            delay_w2mic = round((self.Fs * dist) / self.c)
            setattr(self, f'{wall_id}_to_mic_delay', delay_w2mic)

            self.node_to_mic[key] = deque([0.0] * delay_w2mic, maxlen=delay_w2mic)

        # NODE to NODE DELAY LINES
        for wall1_id, wall1 in self.room.walls.items():
            self.node_to_node[wall1_id] = {}
            for wall2_id, wall2 in self.room.walls.items():
                if wall1_id != wall2_id:
                    dist = wall1.node_positions.getDistance(wall2.node_positions)
                    delay_samples = round((self.Fs * dist) / self.c)

                    self.node_to_node[wall1_id][wall2_id] = deque([0.0] * delay_samples, maxlen=delay_samples)
                    setattr(self, f'{wall1_id}_to_{wall2_id}_delay', delay_samples)

    def get_delay_by_path(self, path):
        """
        Calculate the total delay for a given path by summing up individual segment delays.
        Args:
            path (list): A list of node IDs representing the path, e.g., ['src', 'west', 'floor', 'mic']
        Returns:
            int: The total delay in samples for the complete path
        """
        total_delay = 0
        # Iterate through the path segments
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            # Handle the special case for source-to-node delay
            if start_node == 'src':
                delay_attr_name = f'src_to_{end_node}_delay'
            # Handle the special case for node-to-mic delay
            elif end_node == 'mic':
                delay_attr_name = f'{start_node}_to_mic_delay'
            # Handle node-to-node delay
            else:
                delay_attr_name = f'{start_node}_to_{end_node}_delay'
            
            # Get the delay value from the DelayNetwork instance
            try:
                segment_delay = getattr(self, delay_attr_name)
                # Add to the total delay
                total_delay += segment_delay
            except AttributeError:
                # If the attribute doesn't exist, use a default delay
                print(f"Warning: Delay attribute {delay_attr_name} not found. Using default delay of 0.")
                segment_delay = 0
                
        return total_delay

    def _update_scattering_matrix(self):
        """Update the scattering matrix based on current settings."""
        N = self.num_nodes
        size = N - 1  # Matrix size for each node's operation

        if self.use_identity_scattering:
            self.scattering_matrix = np.eye(size)
        elif self.specular_scattering:
            self.scattering_matrix = ssm.build_specular_matrices()
        else:
            # Default diffuse, original scattering
            self.scattering_matrix = (2 / size) * np.ones((size, size)) - np.eye(size)

    def process_sample(self, input_sample, n) -> float:
        """Process one sample through the network and return the output sample."""
        output_sample = 0.0
        self.n = n

        # Step 0a : propagate the DIRECT SOUND to source_to_mic delay line
        direct_sound = input_sample * self.source_to_mic_gain

        self.source_to_mic["src_to_mic"].append(direct_sound)
        output_sample += self.source_to_mic["src_to_mic"][0]

        # Step 1: Distribute source signal
        if self.ho_sdn_order is not None:
            # a) Inject into early reflection delay lines (direct to mic)
            for key, gain in self.early_reflection_gains.items():
                early_pressure = input_sample * gain
                self.early_reflection_lines[key].append(early_pressure)

            # b) Inject into SDN-feeding delay lines (order N)
            for key, gain in self.ho_source_to_node_gains.items():
                source_pressure = input_sample * gain
                self.ho_source_to_nodes[key].append(source_pressure)
        else:
            # Standard SDN: Inject into first-order source-to-node lines
            for wall_id in self.room.walls:
                src_key = f"src_to_{wall_id}"
                source_pressure = input_sample * self.source_to_node_gains[src_key]
                self.source_to_nodes[src_key].append(source_pressure)

        # Step 1b: Collect output from early reflection lines
        if self.ho_sdn_order is not None:
            for key in self.early_reflection_lines:
                output_sample += self.early_reflection_lines[key][0]

        # Step 2: Process each node
        for wall_id in self.room.walls:
            # Step 2a: Read arriving source pressure from source_to_node delay line(s)
            source_pressure_at_wall = 0.0
            if self.ho_sdn_order is not None:
                 # Sum pressure from all order-N paths arriving at this wall
                 for i, img_info in enumerate(self.sdn_feeding_paths):
                     if img_info['path'][-1] == wall_id:
                         key = f"sdn_feed_path_{i}_to_{wall_id}"
                         source_pressure_at_wall += self.ho_source_to_nodes[key][0]
            else:
                # Standard SDN: Read from single first-order path
                src_key = f"src_to_{wall_id}"
                source_pressure_at_wall = self.source_to_nodes[src_key][0]

            # Collect incoming wave variables
            incoming_waves = []
            other_nodes = []
            psk = self.source_pressure_injection_coeff * source_pressure_at_wall  # source pressure arriving at any (all) nodes (mostly zero except impulse sample)

            # Get both best and second-best targets
            targets = get_best_reflection_targets(wall_id, self.room.angle_mappings, num_targets=2)
            best_target = targets[0]
            if self.specular_source_injection_random:
                best_target = self.random_target_map[wall_id]
                # print(f"Best target for wall {wall_id} is randomly mapped to: {best_target}")
            second_best_target = targets[1]
            # target = get_best_reflection_target(wall_id, self.room.angle_mappings) # old
            iter = 1
            psk_in = []  # Initialize pin for each wall_id
            for other_id in self.room.walls:
                if other_id != wall_id:
                    other_nodes.append(other_id)  # Add this line to collect other_nodes
                    # Read from delay line and add half of source pressure

                    pki_pressure = self.node_to_node[other_id][wall_id][0]  # incoming wave from other node at n

                    if self.specular_source_injection or self.specular_source_injection_random:
                        if psk != 0.0: # change the source injection distribution. new approach.

                            if self.injection_c_vector is not None:
                                c = self.injection_c_vector[self.injection_index]
                            elif self.injection_vector is not None:
                                c = self.injection_vector[0]  # First element always for dominant node
                            else:
                                c = self.source_weighting

                            # print("wall_id:", wall_id, "other_id:", other_id)
                            if other_id == best_target:
                                # print(iter, "dom:", c)
                                if self.print_weighted_psk:
                                    print("best_target:", best_target)
                                psk_in.append(c)
                                p_tilde = pki_pressure + c*psk

                            else:
                                # Only calculate cn = (5-c)/4 if we're not using injection_vector
                                if self.injection_vector is not None:
                                    # Use non_dominant_index to get values from the injection vector
                                    if self.non_dominant_index < len(self.injection_vector):
                                        cn = self.injection_vector[self.non_dominant_index]
                                        self.non_dominant_index += 1  # Increment for next non-dominant node
                                else:
                                    # Original calculation only used when no injection_vector
                                    cn = (5 - c) / 4
                                
                                psk_in.append(cn)
                                p_tilde = pki_pressure + cn * psk # or pki_pressure + 0 * psk
                            
                            # Check total injection count at the end of the loop
                            # only used if there is injection_vector?
                            if iter == self.total_injection_count-1:
                                self.injection_index += 1
                                self.non_dominant_index = 1  # Reset for next round of source injection
                                if self.print_weighted_psk:
                                    print("weighted psk vector", psk_in)
                            
                            iter += 1

                        else: # if source pressure = 0, no need to adjust source injection
                            p_tilde = pki_pressure # since psk=0

                        incoming_waves.append(p_tilde) # p_tilde is calculated according to above if-else's.

                    elif self.source_trial_injection is not None:
                        p_tilde = pki_pressure + self.source_trial_injection * psk
                        incoming_waves.append(p_tilde)

                    else: # original SDN , no change for psk distribution
                        p_tilde = pki_pressure + psk  # if p_tilde is zero, no pressure at the node
                        incoming_waves.append(p_tilde)

            # Apply appropriate scattering matrix
            if self.specular_scattering:
                self.instant_outgoing_waves = np.dot(self.scattering_matrix[wall_id], incoming_waves)
                
            elif self.specular_node_pressure:
                
                # if there is one abs(max) pressure in incoming_waves, find the index of it
                max_incoming_node_index = np.argmax(np.abs(incoming_waves))
                max_incoming_node_id = other_nodes[max_incoming_node_index]
                targets = get_best_node2node_targets(wall_id, max_incoming_node_id, self.room.angle_mappings,
                                                        num_targets=2)
                best_specular_node_target = targets[0]
                #take the index of other_nodes[best_specular_node_target]
                best_specular_node_target_index = other_nodes.index(best_specular_node_target)
                if max_incoming_node_index != best_specular_node_target_index:
                    self.scattering_matrix = swap(self.scattering_matrix, max_incoming_node_index,
                                                    best_specular_node_target_index)

                    self.instant_outgoing_waves = np.dot(self.scattering_matrix, incoming_waves)
                    # Swap back to restore the original matrix state
                    self.scattering_matrix = swap(self.scattering_matrix, max_incoming_node_index, best_specular_node_target_index)
                else:
                    self.instant_outgoing_waves = np.dot(self.scattering_matrix, incoming_waves)

            else: # Use default scattering matrix (identity or diffuse)
                self.instant_outgoing_waves = np.dot(self.scattering_matrix, incoming_waves)

            # Create diagonal attenuation matrix with wall-specific attenuation values
            attenuation_values = [self.room.wallAttenuation[self.room.walls[other_id].wall_index] for other_id in other_nodes]
            att = np.diag(attenuation_values)
            # Apply attenuation matrix to the scattered waves
            self.instant_outgoing_waves = np.dot(att, self.instant_outgoing_waves)

            all_zero = all(element == 0 for element in self.instant_outgoing_waves)

            # Store outgoing waves for each connection
            for idx, other_id in enumerate(other_nodes):

                self.outgoing_waves[wall_id][other_id] = self.instant_outgoing_waves[idx]

            # Calculate node pressure : p_k(n) =  2/(N-1) * Σ p_ki^+(n)+ p_Sk(n)/2
            node_pressure = self.coef * sum(self.instant_outgoing_waves)
            # node_pressure = self.coef * sum(incoming_waves)
            # self.node_pressures[wall_id] = node_pressure # kullanılmıyor ?

            # Send to microphone (using outgoing waves)
            mic_key = f"{wall_id}_to_mic"

            # coef = 2/5 originally
            # what about using? sum(incoming_waves)
            mic_pressure = node_pressure * \
                           self.node_to_mic_gains[mic_key]  # \
                           # * self.room.wallAttenuation[self.room.walls[wall_id].wall_index]
            if self.print_mic_pressures:
                if mic_pressure != 0.0:  # print when mic pressure is nonzero
                    print("mic_pressure:", mic_pressure*5/2)

            self.node_to_mic[mic_key].append(mic_pressure)

            output_sample += self.node_to_mic[mic_key][0]

        # Step 3: Update node-to-node connections using stored outgoing waves
        for wall_id in self.room.walls:
            for other_id in self.room.walls:
                if wall_id != other_id:
                    # Get outgoing wave and apply wall attenuations
                    outgoing_wave = self.outgoing_waves[wall_id][other_id]
                    # sending_wall_atten = self.room.wallAttenuation[self.room.walls[wall_id].wall_index]

                    # attenuated_wave = outgoing_wave * sending_wall_atten
                    attenuated_wave = outgoing_wave

                    self.node_to_node[wall_id][other_id].append(attenuated_wave)

        return output_sample

    def _calculate_attenuations(self):
        """Calculate attenuation factors for all connections."""
        G = self.c / self.Fs  # unit distance

        # Direct sound attenuation (1/r law)
        src_mic_distance = self.room.source.srcPos.getDistance(self.room.micPos)
        self.source_to_mic_gain = G / src_mic_distance if not self.ignore_src_node_atten else 1.0

        if self.ho_sdn_order is not None:
            # --- HO-SDN Attenuation Scheme ---
            self.ho_source_to_node_gains = {}
            self.node_to_mic_gains = {}
            self.early_reflection_gains = {}

            # 1. Attenuation for early reflections (order < N), direct to mic
            # Gain is 1/distance
            for i, path_info in enumerate(self.early_reflection_paths):
                dist = path_info['position'].getDistance(self.room.micPos)
                key = f"early_path_{i}"
                self.early_reflection_gains[key] = (G / dist) if not self.ignore_src_node_atten else 1.0


            # 2. Attenuation for SDN-feeding paths (order N)
            for i, img_info in enumerate(self.sdn_feeding_paths):
                last_wall_id = img_info['path'][-1]
                node_pos = self.room.walls[last_wall_id].node_positions
                
                alpha = node_pos.getDistance(img_info['position'])
                beta = node_pos.getDistance(self.room.micPos)

                key = f"sdn_feed_path_{i}_to_{last_wall_id}"
                # Formula from analysis of Simulation_HO_SDN_centroid.py
                # Total gain = G / (alpha + beta).
                # Here we use the g_sk = 1/(1+alpha/beta) split from the paper, but multiply by G
                # to match the reference implementation's spherical spreading loss model.
                self.ho_source_to_node_gains[key] = (G / (1.0 + alpha / beta)) if not self.ignore_src_node_atten else 1.0

            # 3. Node-to-mic gain (one per node), g_kr = 1/beta
            # The reference code uses 1/beta, not G/beta. We follow that.
            for wall_id, wall in self.room.walls.items():
                mic_key = f"{wall_id}_to_mic"
                beta = wall.node_positions.getDistance(self.room.micPos)
                self.node_to_mic_gains[mic_key] = (1.0 / beta) if not self.ignore_node_mic_atten else 1.0

        else:
            # --- Standard SDN Attenuation Scheme ---
            self.source_to_node_gains = {}
            self.node_to_mic_gains = {}
            for wall_id, wall in self.room.walls.items():
                node_pos = wall.node_positions
                src_dist = node_pos.getDistance(self.room.source.srcPos)
                mic_dist = node_pos.getDistance(self.room.micPos)

                # Source to node: g_sk = G/||x_S - x_k||
                src_key = f"src_to_{wall_id}"
                self.source_to_node_gains[src_key] = G / src_dist if not self.ignore_src_node_atten else 1.0

                # Node to mic: g_km = 1/(1 + ||x_k - x_M||/||x_S - x_k||)
                mic_key = f"{wall_id}_to_mic"
                self.node_to_mic_gains[mic_key] = 1.0 / (
                            1.0 + mic_dist / src_dist) if not self.ignore_node_mic_atten else 1.0

    def calculate_rir(self, duration):
        """Calculate room impulse response for a given duration."""
        num_samples = int(self.Fs * duration)
        rir = np.zeros(num_samples)

        for n in range(num_samples):
            rir[n] = self.process_sample(self.room.source.signal[n], n)

        return rir

    def plot_wall_incoming_sums(self):
        """Plot incoming pressure sums for each wall."""
        plt.figure(figsize=(12, 6))
        for wall_id, sums in self.wall_incoming_sums.items():
            plt.scatter(self.sample_indices[wall_id], sums, label=f'{wall_id}')
            plt.plot(self.sample_indices[wall_id], sums, linestyle='-', alpha=0.5)  # Connect the points with a line
        plt.title('Incoming Pressure Sums at Each Wall')
        plt.xlabel('Sample Index (n)')
        plt.ylabel('Sum of Incoming Pressures')
        plt.grid(True)
        plt.legend()
        plt.show()
        return