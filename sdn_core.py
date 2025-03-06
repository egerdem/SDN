from collections import deque
import numpy as np
from typing import Dict, List, Optional
from geometry import Room, Point, build_angle_mappings, get_best_reflection_target, build_specular_matrices_from_angles
from path_logger import PathLogger, PressurePacket, deque_plotter
import matplotlib.pyplot as plt
import path_logger as pl
import specular_scattering_matrix as ssm
import random

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


class DelayNetwork:
    """Core SDN implementation focusing on sample-based processing with accessible delay lines."""

    def __init__(self, room: Room, Fs: int = 44100, c: float = 343.0, source_pressure_injection_coeff: float = 0.5,
                 coef: float = 2.0/5,
                 source_weighting: float = 1,
                 use_identity_scattering: bool = False,
                 specular_scattering: bool = False,
                 specular_source_injection: bool = False,
                 ignore_wall_absorption: bool = False,
                 ignore_src_node_atten: bool = False,
                 ignore_node_mic_atten: bool = False,
                 enable_path_logging: bool = False,
                 scattering_matrix_update_coef: float = None,
                 more_absorption: bool = False,
                 print_mic_pressures: bool = False,
                 print_parameter_summary: bool = False,
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
            enable_path_logging: If True, enable detailed path logging (default: False)
            scattering_matrix_update_coef: If not None, use modified scattering matrix with this coefficient
        """
        self.room = room
        self.Fs = Fs
        self.c = c
        self.source_pressure_injection_coeff = source_pressure_injection_coeff
        self.coef = coef
        self.source_weighting = source_weighting

        # Test flags
        self.use_identity_scattering = use_identity_scattering
        self.specular_scattering = specular_scattering
        self.specular_source_injection = specular_source_injection
        self.ignore_wall_absorption = ignore_wall_absorption
        self.ignore_src_node_atten = ignore_src_node_atten
        self.ignore_node_mic_atten = ignore_node_mic_atten
        self.more_absorption = more_absorption
        self.scattering_matrix_update_coef = scattering_matrix_update_coef
        self.print_mic_pressures = print_mic_pressures
        self.print_parameter_summary = print_parameter_summary
        # Override wall attenuation if ignore_wall_absorption is True
        if self.ignore_wall_absorption:
            self.room.wallAttenuation = [1.0] * len(self.room.wallAttenuation)

        # Initialize path logger if enabled
        self.enable_path_logging = enable_path_logging
        self.path_logger = PathLogger() if enable_path_logging else None
        self.label = label

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
        self._setup_delay_lines()

        # Initialize attenuation factors with matching keys
        # self.source_to_mic_gain is added inside the _calculate_attenuations() function
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
            non_default_params.append("• Using Specular Source Injection")
            if self.source_weighting != defaults['source_weighting']:
                non_default_params.append(f"  - Source Weighting Factor: {self.source_weighting}")

        # Attenuation Settings
        if self.ignore_wall_absorption:
            non_default_params.append("• Wall Absorption Ignored")
        if self.ignore_src_node_atten:
            non_default_params.append("• Source-Node Attenuation Ignored")
        if self.ignore_node_mic_atten:
            non_default_params.append("• Node-Mic Attenuation Ignored")

        # Path Logging
        if self.enable_path_logging:
            non_default_params.append("• Path Logging Enabled")

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
            return build_specular_matrices_from_angles(self.room)
        elif self.scattering_matrix_update_coef is not None:
            print(scattering_matrix_crate(self.scattering_matrix_update_coef))
            return scattering_matrix_crate(self.scattering_matrix_update_coef)
        else:
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


        if self.enable_path_logging == False: # NO LOG CASE - REF
            self.source_to_mic[key] = deque([0.0] * self.direct_sound_delay, maxlen=self.direct_sound_delay)

        ###############################################
        else: # LOG CASE
            initial_packet = PressurePacket(value=0.0, path_history=['src', 'mic'], birth_sample=0,
                                        delay=self.direct_sound_delay)
            self.source_to_mic[key] = deque([initial_packet] * self.direct_sound_delay, maxlen=initial_packet.delay)
            # deque_plotter(self.source_to_mic[key])
        ###############################################

        # SOURCE to NODE and NODE to MIC DELAY LINES
        for wall_id, wall in self.room.walls.items():

            # Source to nodes
            key = f"src_to_{wall_id}"
            dist = wall.node_positions.getDistance(self.room.source.srcPos)
            delay_src2node = round((self.Fs * dist) / self.c)
            self.source_to_nodes[key] = deque([0.0] * delay_src2node, maxlen=delay_src2node)
            setattr(self, f'src_to_{wall_id}_delay', delay_src2node)

            # Nodes to microphone
            key = f"{wall_id}_to_mic"
            dist = wall.node_positions.getDistance(self.room.micPos)
            delay_w2mic = round((self.Fs * dist) / self.c)
            setattr(self, f'{wall_id}_to_mic_delay', delay_w2mic)

            ###############################################
            if self.enable_path_logging:
                initial_packet = PressurePacket(value=0.0, path_history=[wall_id, 'mic'], birth_sample=0,
                                                delay=delay_w2mic)
                self.node_to_mic[key] = deque([initial_packet] * delay_w2mic, maxlen=delay_w2mic)
            ###############################################

            else:
                self.node_to_mic[key] = deque([0.0] * delay_w2mic, maxlen=delay_w2mic)

        # NODE to NODE DELAY LINES
        for wall1_id, wall1 in self.room.walls.items():
            self.node_to_node[wall1_id] = {}
            for wall2_id, wall2 in self.room.walls.items():
                if wall1_id != wall2_id:
                    dist = wall1.node_positions.getDistance(wall2.node_positions)
                    delay_samples = round((self.Fs * dist) / self.c)

                    if self.enable_path_logging:
                        initial_packet = PressurePacket(value=0.0, path_history=[wall1_id, wall2_id], birth_sample=0, delay=delay_samples)
                        self.node_to_node[wall1_id][wall2_id] = deque([initial_packet] * delay_samples, maxlen=delay_samples)
                        # self.node_to_node[wall1_id][wall2_id] = deque([0.0] * delay_samples, maxlen=delay_samples)
                        setattr(self, f'{wall1_id}_to_{wall2_id}_delay', delay_samples)

                    else:  # NO LOG CASE - REF
                        self.node_to_node[wall1_id][wall2_id] = deque([0.0] * delay_samples, maxlen=delay_samples)

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

        if self.enable_path_logging == False:  # NO LOG CASE - REF
            self.source_to_mic["src_to_mic"].append(direct_sound)
            output_sample += self.source_to_mic["src_to_mic"][0]

        ###############################################
        else:  # LOG CASE
            direct_packet = PressurePacket(
                value=direct_sound,
                path_history=['src', 'mic'],
                birth_sample=n,
                delay=self.direct_sound_delay)
            self.source_to_mic["src_to_mic"].append(direct_packet)
            output_sample += self.source_to_mic["src_to_mic"][0].value

            if abs(direct_sound) > 1e-10:
                self.path_logger.log_packet(direct_packet)
        ###############################################


        # Step 1: Distribute source to nodes with gains
        for wall_id in self.room.walls:
            src_key = f"src_to_{wall_id}"
            source_pressure = input_sample * self.source_to_node_gains[src_key] #input sample is the source signal 1,0,0... at n

            if self.enable_path_logging == False: # NO LOG CASE - REF
                self.source_to_nodes[src_key].append(source_pressure)

            ###############################################
            else: # LOG CASE
                if source_pressure == 0:  # no source input, Just append zero and move the delay line forward
                    assert source_pressure == 0.0, f"source_pressure: {source_pressure}, should be zero!"
                    self.source_to_nodes[src_key].append(0.0)

                else:  # nonzero source input: source reaches a node: append the packet
                    assert source_pressure != 0.0, f"source_pressure: {source_pressure}, should be nonzero!"
                    # Create the packet for the delay line
                    packet = PressurePacket(
                        value=source_pressure,
                        path_history=['src', wall_id],
                        birth_sample=n,
                        delay=getattr(self, f'src_to_{wall_id}_delay'),
                    )
                    self.source_to_nodes[src_key].append(packet)
                    
                    # We don't log this packet here because it's not a complete path
                    # It will be logged when it reaches the microphone
                    # This prevents logging incomplete paths
            ###############################################

        # Step 2: Process each node
        for wall_id in self.room.walls:
            src_key = f"src_to_{wall_id}"

            # Step 2a: Read arriving source pressure from source_to_node delay line
            if self.enable_path_logging==False: # NO LOG CASE - REF
                source_pressure = self.source_to_nodes[src_key][0]

            ###############################################
            else: # LOG CASE
                if isinstance(self.source_to_nodes[src_key][0], PressurePacket): # if there is a nonzero source arrival : packet exist
                    source_pressure = self.source_to_nodes[src_key][0].value
                else:  # there is no active source input, Just append zero and move the delay line forward
                    source_pressure = self.source_to_nodes[src_key][0]
                    assert source_pressure == 0.0, f"source_pressure: {source_pressure}, should be zero!"
            ###############################################

            # Collect incoming wave variables
            incoming_waves = []
            other_nodes = []
            incoming_packets = []  # Store packets for path tracking

            psk = self.source_pressure_injection_coeff * source_pressure  # source pressure arriving at any (all) nodes (mostly zero except impulse sample)
            # if psk != 0.0 or psk != 0: # print when source pressure is nonzero (happens only 6 times as there are 6 nodes)
            #     print("psk:", psk)

            iter = 0

            target = get_best_reflection_target(wall_id, self.room.angle_mappings)

            # print("Wall:", wall_id)
            # target = random.choice(list(self.room.walls.keys()))
            # print("target:", target)
            i = 1
            for other_id in self.room.walls:
                if other_id != wall_id:
                    other_nodes.append(other_id)
                    # Read from delay line and add half of source pressure

                    if self.enable_path_logging == False:
                        pki_pressure = self.node_to_node[other_id][wall_id][0]  # incoming wave from other node at n

                    ###############################################
                    else:  # LOG CASE
                        packet = self.node_to_node[other_id][wall_id][0]
                        pki_pressure = packet.value
                        
                        # Only extend the path if the packet has a non-empty path history
                        # and non-zero pressure
                        if abs(pki_pressure) > 1e-10 and packet.path_history:
                            # DO NOT extend the path here - we're just reading from the delay line
                            # The path will be extended when we update the node-to-node connections
                            # This prevents duplicate wall IDs in the path history
                            incoming_packets.append(packet)  # Use the original packet without extending
                        else:
                            # For zero pressure or empty path history, create a placeholder packet
                            # This ensures we have something in incoming_packets for each other_id
                            placeholder_packet = PressurePacket(
                                value=pki_pressure,
                                path_history=[],
                                birth_sample=n,
                                delay=0
                            )
                            incoming_packets.append(placeholder_packet)

                    ###############################################

                    p_tilde = pki_pressure + psk  # if p_tilde is zero, no pressure at the node

                    if self.specular_source_injection:
                        c = self.source_weighting
                        if other_id == target:
                            incoming_waves.append(pki_pressure + c*psk)
                        else:
                            incoming_waves.append(pki_pressure + (5 - c) / 4 * psk)
                            # incoming_waves.append(pki_pressure + 0 * psk)
                    else:
                        incoming_waves.append(p_tilde)

            # Apply appropriate scattering matrix
            if self.specular_scattering:
                self.instant_outgoing_waves = np.dot(self.scattering_matrix[wall_id], incoming_waves)
            else:
                # Use default scattering matrix (identity or diffuse)
                self.instant_outgoing_waves = np.dot(self.scattering_matrix, incoming_waves)

            all_zero = all(element == 0 for element in self.instant_outgoing_waves)

            # Store outgoing waves for each connection
            for idx, other_id in enumerate(other_nodes):
                val = self.instant_outgoing_waves[idx]

                if self.enable_path_logging == False:  # NO LOG CASE - REF
                    self.outgoing_waves[wall_id][other_id] = val

                ###############################################
                else: # LOG CASE
                    if all_zero == False:  # if there are outgoing pressures at the node, we should store
                        # Get the corresponding incoming packet for this outgoing wave
                        incoming_packet = incoming_packets[idx]
                        
                        # Create new packet for scattered wave with proper path history
                        # We're just maintaining the path history, not logging it yet
                        scattered_packet = PressurePacket(
                            value=val,
                            path_history=incoming_packet.path_history,  # Path already includes wall_id from extend_path
                            birth_sample=n,
                            delay=incoming_packet.delay  # Maintain the accumulated delay
                        )
                        
                        self.outgoing_waves[wall_id][other_id] = scattered_packet
                        
                        # We don't log this packet here because it's not a complete path
                        # It will be logged when it reaches the microphone

                    else:
                        val = 0.0
                        scattered_packet = PressurePacket(
                            value=val,
                            path_history=[],  # Empty path for zero pressure
                            birth_sample=n,
                            delay=0
                        )

                        self.outgoing_waves[wall_id][other_id] = scattered_packet
                        assert all_zero, f"outgoingler 0 olmalıydı"  # there should be no pressure at the node
                ###############################################

            # Calculate node pressure : p_k(n) =  2/(N-1) * Σ p_ki^+(n)+ p_Sk(n)/2
            node_pressure = self.coef * sum(self.instant_outgoing_waves)
            # self.node_pressures[wall_id] = node_pressure # kullanılmıyor ?

            # Send to microphone (using outgoing waves)
            mic_key = f"{wall_id}_to_mic"

            # coef = 2/5 originally
            # what about using? sum(incoming_waves)
            mic_pressure = node_pressure * \
                           self.node_to_mic_gains[mic_key] * \
                           self.room.wallAttenuation[self.room.walls[wall_id].wall_index]
            if self.print_mic_pressures:
                if mic_pressure != 0.0:  # print when mic pressure is nonzero
                    print("mic_pressure:", mic_pressure*5/2)

            if self.enable_path_logging == False: # NO LOG CASE - REF
                self.node_to_mic[mic_key].append(mic_pressure)

            ###############################################
            else: # LOG CASE
                # For each node, we need to track the complete path from source to mic
                # First, check if there's any non-zero pressure at this node
                if abs(node_pressure) > 1e-10:
                    # Reconstruct the complete path
                    # We need to find which incoming packets contributed to this node's pressure
                    contributing_paths = []
                    
                    # Check all incoming packets for non-zero contributions
                    for packet in incoming_packets:
                        if abs(packet.value) > 1e-10:
                            # This packet contributed to the node pressure
                            # Make sure the path starts with 'src'
                            path = packet.path_history.copy()  # Make a copy to avoid modifying the original
                            if path and path[0] != 'src':
                                # If path doesn't start with src, prepend it
                                path = ['src'] + path
                            
                            # Only add complete paths that start with 'src'
                            if path and path[0] == 'src':
                                # Check if the current wall_id is already the last node in the path
                                if not path or path[-1] != wall_id:
                                    path.append(wall_id)
                                
                                # Add the path to contributing paths
                                contributing_paths.append(path)
                    
                    # If we have contributing paths, create a mic packet with the full path
                    if contributing_paths:
                        for path in contributing_paths:
                            # Create packet for microphone output with complete path
                            # Make sure we don't add 'mic' if it's already there
                            if path and path[-1] != 'mic':
                                path = path + ['mic']
                                
                            mic_packet = PressurePacket(
                                value=mic_pressure / len(contributing_paths),  # Distribute pressure among paths
                                path_history=path,  # Complete path from src to mic
                                birth_sample=n,
                                delay=getattr(self, f'{wall_id}_to_mic_delay')
                            )
                            
                            # Log the complete path
                            if abs(mic_pressure) > 1e-10:
                                self.path_logger.log_packet(mic_packet)
                    else:
                        # Fallback if no contributing paths found - only use this for direct source-to-node-to-mic paths
                        # This should only happen for first-order reflections
                        mic_packet = PressurePacket(
                            value=mic_pressure,
                            path_history=['src', wall_id, 'mic'],
                            birth_sample=n,
                            delay=getattr(self, f'{wall_id}_to_mic_delay')
                        )
                        if abs(mic_pressure) > 1e-10:
                            self.path_logger.log_packet(mic_packet)
                
                # Always append something to the delay line - but don't log this
                # This is just for the delay line, not for path tracking
                default_packet = PressurePacket(
                    value=mic_pressure,
                    path_history=['src', wall_id, 'mic'],  # Simple path for delay line
                    birth_sample=n,
                    delay=getattr(self, f'{wall_id}_to_mic_delay')
                )
                self.node_to_mic[mic_key].append(default_packet if abs(mic_pressure) > 1e-10 else 0.0)
            ###############################################
            output_sample += self.node_to_mic[mic_key][0] if not isinstance(self.node_to_mic[mic_key][0],
                                                                            PressurePacket) else self.node_to_mic[mic_key][0].value

        # Step 3: Update node-to-node connections using stored outgoing waves
        for wall_id in self.room.walls:
            for other_id in self.room.walls:
                if wall_id != other_id:
                    # Get outgoing wave and apply wall attenuations
                    outgoing_wave = self.outgoing_waves[wall_id][other_id]
                    sending_wall_atten = self.room.wallAttenuation[self.room.walls[wall_id].wall_index]

                    if self.enable_path_logging == False: # NO LOG CASE - REF
                        attenuated_wave = outgoing_wave * sending_wall_atten
                        self.node_to_node[wall_id][other_id].append(attenuated_wave)

                    ###############################################
                    else: # LOG CASE
                        if isinstance(outgoing_wave, PressurePacket):
                            # Create a new packet with the attenuated value and proper path history
                            # We need to extend the path with the current node and target node
                            # But first check if the path already ends with the current node
                            path_history = outgoing_wave.path_history
                            
                            # Only add the current node (wall_id) if it's not already the last node in the path
                            if not path_history or path_history[-1] != wall_id:
                                path_history = path_history + [wall_id]
                                
                            # Now add the target node (other_id)
                            path_history = path_history + [other_id]
                            
                            attenuated_packet = PressurePacket(
                                value=outgoing_wave.value * sending_wall_atten,
                                path_history=path_history,  # Updated path history
                                birth_sample=outgoing_wave.birth_sample,
                                delay=outgoing_wave.delay + getattr(self, f'{wall_id}_to_{other_id}_delay', 0)  # Add delay
                            )
                            self.node_to_node[wall_id][other_id].append(attenuated_packet)
                        else:
                            # Handle the case where outgoing_wave is not a PressurePacket (should be 0.0)
                            zero_packet = PressurePacket(
                                value=0.0,
                                path_history=[],  # Empty path for zero pressure
                                birth_sample=n,
                                delay=0
                            )
                            self.node_to_node[wall_id][other_id].append(zero_packet)
                    ###############################################

        # self.current_sample += 1
        return output_sample

    def _calculate_attenuations(self):
        """Calculate attenuation factors for all connections."""
        G = self.c / self.Fs  # unit distance

        # Direct sound attenuation (1/r law)
        src_mic_distance = self.room.source.srcPos.getDistance(self.room.micPos)
        self.source_to_mic_gain = G / src_mic_distance if not self.ignore_src_node_atten else 1.0

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

    def get_path_summary(self, path_key: Optional[str] = None):
        """Get summary of logged paths."""
        if not self.enable_path_logging:
            return "Path logging is not enabled. Initialize DelayNetwork with enable_path_logging=True"

        self.path_logger.print_path_summary(path_key)

    def get_paths_through_node(self, node_id: str):
        """Get all paths passing through a specific node."""
        if not self.enable_path_logging:
            return "Path logging is not enabled. Initialize DelayNetwork with enable_path_logging=True"

        return self.path_logger.get_paths_through_node(node_id)

    def get_active_paths_at_sample(self, sample_idx: int):
        """Get all active paths at a specific sample index."""
        if not self.enable_path_logging:
            return "Path logging is not enabled. Initialize DelayNetwork with enable_path_logging=True"

        return self.path_logger.get_active_paths_at_sample(sample_idx)

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