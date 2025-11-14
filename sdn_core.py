from collections import deque
import numpy as np
from typing import Dict, List, Optional
from geometry import Room, get_best_node2node_targets, get_best_reflection_targets, build_specular_matrices_from_angles, get_image_sources
import matplotlib.pyplot as plt
from research import specular_scattering_matrix as ssm


def swap(s, j, k):
    """Swap two rows in a matrix."""
    s_jk = s.copy()
    s_jk[[j, k]] = s_jk[[k, j]]
    return s_jk

class DelayNetwork:
    """Core SDN implementation focusing on sample-based processing
    There are many test flags to modify the behavior of the SDN for research purposes.
    Some of them are promising but left unexplored due to time constraints. Ask Ege if you need more info.
    """

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
            self.random_target_map = ssm.random_wall_mapping(list(self.room.walls.keys()))
            print("Random wall mapping created:")
            print(self.random_target_map)

        # Print parameter summary
        if self.print_parameter_summary:
            self._print_parameter_summary()

        # SDN parameters
        self.num_nodes = len(room.walls)  # Number of scattering nodes (6 for cuboid)

        # Initialize scattering matrices
        self.scattering_matrix = self._create_scattering_matrix()

        # ============================================================================
        # DELAY LINE DATA STRUCTURES (deques storing propagating samples)
        # ============================================================================
        # All delay lines are dictionaries containing deques (double-ended queues)
        # Deques act as circular buffers: append at right, pop from left (FIFO)
        
        # Direct path (0th order)
        self.source_to_mic = {}  # Dict with single key: "src_to_mic" -> deque
        
        # Standard SDN delay lines (1st order reflections and beyond)
        self.source_to_nodes = {}  # Dict: "src_to_{wall_id}" -> deque (e.g. "src_to_floor")
        self.node_to_mic = {}      # Dict: "{wall_id}_to_mic" -> deque (e.g. "floor_to_mic")
        self.node_to_node = {}     # Nested dict: {wall1_id: {wall2_id: deque}} (e.g. ["floor"]["ceiling"])

        if self.ho_sdn_order is not None:
            print(f"--- Initializing HO-SDN of order {self.ho_sdn_order} ---")
            all_image_sources = get_image_sources(self.room, self.ho_sdn_order)
            print("all_image_sources:", len(all_image_sources))
            # Early reflections (order < N) are handled as direct paths to the mic
            self.early_reflection_paths = [img for img in all_image_sources if img['order'] < self.ho_sdn_order and img['order'] > 0]
            self.early_reflection_del_lines = {} # "path_idx" -> deque
            
            # Order-N reflections feed the SDN network
            self.sdn_feeding_paths = [img for img in all_image_sources if img['order'] == self.ho_sdn_order]
            self.ho_source_to_nodes = {}  # Format: "src_img_{img_idx}_to_{wall_id}"
        else:
            self.early_reflection_paths = []
            self.sdn_feeding_paths = []
            self.ho_source_to_nodes = None

        self._setup_delay_lines()

        # ============================================================================
        # ATTENUATION/GAIN FACTORS (distance-based amplitude scaling)
        # ============================================================================
        # These are scalar multipliers applied to signals for spherical spreading loss
        
        self.source_to_node_gains = {}  # Dict: "src_to_{wall_id}" -> float (g_sk = G/||x_S - x_k||)
        self.node_to_mic_gains = {}     # Dict: "{wall_id}_to_mic" -> float (g_km formula varies by mode)
        self._calculate_attenuations()

        # ============================================================================
        # STATE VARIABLES (updated each sample during processing)
        # ============================================================================
        
        # Node pressures at current time step
        self.node_pressures = {}  # Dict: {wall_id: float} e.g. {"floor": 0.5, "ceiling": 0.3, ...}
        for wall_id in room.walls:
            self.node_pressures[wall_id] = 0.0

        # Outgoing waves from each node (computed after scattering matrix)
        self.outgoing_waves = {}  # Nested dict: {wall1_id: {wall2_id: float}} e.g. ["floor"]["ceiling"] = 0.2
        for wall1_id in room.walls:
            self.outgoing_waves[wall1_id] = {}
            for wall2_id in room.walls:
                if wall1_id != wall2_id:
                    self.outgoing_waves[wall1_id][wall2_id] = 0.0

        # Debug/analysis storage (not used in core algorithm)
        self.wall_incoming_sums = {}  # Dict: {wall_id: list[float]} - stores history of incoming sums
        self.sample_indices = {}      # Dict: {wall_id: list[int]} - stores sample indices for plotting
        for wall_id in room.walls:
            self.wall_incoming_sums[wall_id] = []
            self.sample_indices[wall_id] = []


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
            print(ssm.scattering_matrix_crate(self.scattering_matrix_update_coef))
            return ssm.scattering_matrix_crate(self.scattering_matrix_update_coef)
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
                self.early_reflection_del_lines[key] = deque([0.0] * delay, maxlen=delay)

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
        """
        Process one sample through the SDN network and return the output sample.
        
        ═══════════════════════════════════════════════════════════════════════════
        FUNDAMENTAL STEPS A SAMPLE GOES THROUGH IN STANDARD SDN:
        ═══════════════════════════════════════════════════════════════════════════
        
        INPUT: input_sample (usually 1.0 for impulse, then 0.0 for rest)
        
        STEP 0: DIRECT SOUND (0th order reflection)
            - input_sample → [gain] → source_to_mic delay line → output_sample
        
        STEP 1: SOURCE INJECTION (distribute to all walls)
            - input_sample → [gain] → source_to_nodes["src_to_floor"] (append to deque)
            - input_sample → [gain] → source_to_nodes["src_to_ceiling"] (append to deque)
            - ... (repeat for all 6 walls)
        
        STEP 2: NODE PROCESSING (for each wall/node, process reflections)
            For each wall (e.g., "floor"):
            
            2a) READ source pressure arriving at this wall:
                - source_pressure_at_wall = source_to_nodes["src_to_floor"][0] (pop left)
                - psk = 0.5 * source_pressure_at_wall (source injection coefficient)
            
            2b) COLLECT incoming waves from other 5 walls:
                - incoming_waves[0] = node_to_node["ceiling"]["floor"][0] + psk
                - incoming_waves[1] = node_to_node["north"]["floor"][0] + psk
                - incoming_waves[2] = node_to_node["south"]["floor"][0] + psk
                - incoming_waves[3] = node_to_node["east"]["floor"][0] + psk
                - incoming_waves[4] = node_to_node["west"]["floor"][0] + psk
                (5-element array, one from each neighboring wall + source pressure)
            
            2c) SCATTER using scattering matrix:
                - outgoing_waves = ScatteringMatrix @ incoming_waves
                - outgoing_waves = [wall attenuation] * outgoing_waves
                (Redistributes energy to all outgoing directions)
            
            2d) STORE outgoing waves for next iteration:
                - outgoing_waves["floor"]["ceiling"] = outgoing_waves[0]
                - outgoing_waves["floor"]["north"] = outgoing_waves[1]
                - ... (stored in self.outgoing_waves, will be used in Step 3)
            
            2e) CALCULATE node pressure and send to microphone:
                - node_pressure = (2/5) * sum(outgoing_waves)
                - mic_pressure = node_pressure * node_to_mic_gains["floor_to_mic"]
                - node_to_mic["floor_to_mic"].append(mic_pressure)
                - output_sample += node_to_mic["floor_to_mic"][0] (pop left)
        
        STEP 3: UPDATE NODE-TO-NODE CONNECTIONS (propagate for next sample)
            - For each wall-to-wall connection, append the stored outgoing wave:
            - node_to_node["floor"]["ceiling"].append(outgoing_waves["floor"]["ceiling"])
            - node_to_node["floor"]["north"].append(outgoing_waves["floor"]["north"])
            - ... (repeat for all 6×5=30 connections)
            
            These will be read in Step 2b of the NEXT sample's iteration.
        
        OUTPUT: output_sample (sum of direct + all wall contributions)
        ═══════════════════════════════════════════════════════════════════════════
        
        Args:
            input_sample: Source signal amplitude at current time step (typically 1, then 0...)
            n: Current sample index
            
        Returns:
            float: Output sample (microphone pressure at current time step)
        """
        output_sample = 0.0
        self.n = n
        # print(f"sample {n}")
        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 0: propagate the DIRECT SOUND (0th order) to source_to_mic delay line
        # ═══════════════════════════════════════════════════════════════════════════
        direct_sound = input_sample * self.source_to_mic_gain
        self.source_to_mic["src_to_mic"].append(direct_sound)
        output_sample += self.source_to_mic["src_to_mic"][0]

        if self.print_mic_pressures:
            if self.source_to_mic["src_to_mic"][0] != 0.0:
                print(f"direct sound pressure: {self.source_to_mic['src_to_mic'][0]}")
        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 1: SOURCE INJECTION - Distribute input (source) to all scattering nodes
        # ═══════════════════════════════════════════════════════════════════════════
        if self.ho_sdn_order is not None:
            # a) Inject into early reflection delay lines (direct to mic)
            for key, gain in self.early_reflection_gains.items():
                early_pressure = input_sample * gain
                self.early_reflection_del_lines[key].append(early_pressure)

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

        # Step 1b: Collect output from early reflection delay lines (HO-SDN only)
        if self.ho_sdn_order is not None:
            for key in self.early_reflection_del_lines:
                output_sample += self.early_reflection_del_lines[key][0]

        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 2: NODE PROCESSING - For each wall, scatter incoming waves
        # ═══════════════════════════════════════════════════════════════════════════
        for wall_id in self.room.walls:
            # ─────────────────────────────────────────────────────────────────────────
            # STEP 2a: Read source pressure arriving at this node (wall) from source_to_node delay line
            # ─────────────────────────────────────────────────────────────────────────
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

            # ─────────────────────────────────────────────────────────────────────────
            # STEP 2b: Collect incoming waves from all other nodes + source pressure
            # ─────────────────────────────────────────────────────────────────────────
            incoming_waves = []  # Will be size (N-1) array, e.g. 5 elements for 6-wall room
            other_nodes = []     # List of other wall IDs (excluding current wall)
            psk = self.source_pressure_injection_coeff * source_pressure_at_wall  # Default: 0.5 * source_pressure, source pressure arriving at any (all) nodes (mostly zero except impulse sample)

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

            # ─────────────────────────────────────────────────────────────────────────
            # STEP 2c: Apply scattering matrix to redistribute energy
            # ─────────────────────────────────────────────────────────────────────────
            # Scattering matrix redistributes incoming energy to outgoing directions
            # Default diffuse: outgoing[i] = (2/(N-1)) * sum(incoming) - incoming[i]
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

            # ─────────────────────────────────────────────────────────────────────────
            # STEP 2d: Store outgoing waves (will be used in Step 3 to update delay lines)
            # ─────────────────────────────────────────────────────────────────────────
            for idx, other_id in enumerate(other_nodes):
                self.outgoing_waves[wall_id][other_id] = self.instant_outgoing_waves[idx]

            # ─────────────────────────────────────────────────────────────────────────
            # STEP 2e: Calculate node pressure and send to microphone
            # ─────────────────────────────────────────────────────────────────────────
            # Node pressure formula: p_k(n) = (2/(N-1)) * Σ outgoing_waves (= Σ incoming waves, p_ki^+(n)+ p_Sk(n)/2))
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

        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 3: UPDATE NODE-TO-NODE DELAY LINES (propagate waves for next iteration)
        # ═══════════════════════════════════════════════════════════════════════════
        # The outgoing waves stored in Step 2d are now appended to delay lines.
        # These will travel through the delay lines and arrive at destination nodes
        # in future samples (determined by delay line length).
        for wall_id in self.room.walls:
            for other_id in self.room.walls:
                if wall_id != other_id:
                    # Get outgoing wave from current wall heading to other wall and apply wall attenuation
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
