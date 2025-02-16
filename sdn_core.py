from collections import deque
import numpy as np
from typing import Dict, List, Optional
from geometry import Room, Point, build_angle_mappings, get_best_reflection_target, build_specular_matrices_from_angles
from path_logger import PathLogger, PressurePacket, deque_plotter
import matplotlib.pyplot as plt
import path_logger as pl
import specular_scattering_matrix as ssm

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
                 plot_wall_incoming_sums: bool = False,
                 scattering_matrix_update_coef: float = None,
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
        self.scattering_matrix_update_coef = scattering_matrix_update_coef
        print("coef ??*", self.scattering_matrix_update_coef)
        # Override wall attenuation if ignore_wall_absorption is True
        if self.ignore_wall_absorption:
            self.room.wallAttenuation = [1.0] * len(self.room.wallAttenuation)

        # Initialize path logger if enabled
        self.enable_path_logging = enable_path_logging
        self.path_logger = PathLogger() if enable_path_logging else None
        self.label = label
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
        # self.current_sample = 0
        self.node_pressures = {wall_id: 0.0 for wall_id in room.walls}

        # For storing outgoing wave variables at each node
        self.outgoing_waves = {wall_id: {} for wall_id in room.walls}
        for wall1_id in room.walls:
            for wall2_id in room.walls:
                if wall1_id != wall2_id:
                    self.outgoing_waves[wall1_id][wall2_id] = 0.0

        self.wall_incoming_sums = {wall_id: [] for wall_id in room.walls}
        self.sample_indices = {wall_id: [] for wall_id in room.walls}  # To store n values for plotting

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
        src_mic_distance = self.room.srcPos.getDistance(self.room.micPos)
        self.direct_sound_delay = int(np.ceil(((self.Fs * src_mic_distance) / self.c)))

        # initial_packet = PressurePacket(value=0.0, path_history=['src', 'mic'], birth_sample=0, delay=self.direct_sound_delay)
        # self.source_to_mic[key] = deque([initial_packet] * self.direct_sound_delay, maxlen=self.direct_sound_delay) # kaldırdım çünkü zeroları paketsiz yapıyorum
        # deque_plotter(self.source_to_mic[key])

        self.source_to_mic[key] = deque([0.0] * self.direct_sound_delay, maxlen=self.direct_sound_delay)

        # Source to nodes
        for wall_id, wall in self.room.walls.items():
            key = f"src_to_{wall_id}"
            dist = wall.node_positions.getDistance(self.room.source.srcPos)
            delay_samples = round((self.Fs * dist) / self.c)

            self.source_to_nodes[key] = deque([0.0] * delay_samples, maxlen=delay_samples)

            if self.enable_path_logging:
                setattr(self, f'src_to_{wall_id}_delay', delay_samples)

        # Nodes to microphone
        for wall_id, wall in self.room.walls.items():
            key = f"{wall_id}_to_mic"
            dist = wall.node_positions.getDistance(self.room.micPos)
            delay_samples = round((self.Fs * dist) / self.c)

            if self.enable_path_logging:
                initial_packet = PressurePacket(value=0.0, path_history=[wall_id, 'mic'], birth_sample=0,
                                                delay=delay_samples)
                self.node_to_mic[key] = deque([initial_packet] * delay_samples, maxlen=delay_samples)
            else:
                self.node_to_mic[key] = deque([0.0] * delay_samples, maxlen=delay_samples)

            if self.enable_path_logging:
                setattr(self, f'{wall_id}_to_mic_delay', delay_samples)

        # Between nodes
        for wall1_id, wall1 in self.room.walls.items():
            self.node_to_node[wall1_id] = {}
            for wall2_id, wall2 in self.room.walls.items():
                if wall1_id != wall2_id:
                    dist = wall1.node_positions.getDistance(wall2.node_positions)
                    delay_samples = round((self.Fs * dist) / self.c)

                    if self.enable_path_logging:
                        # initial_packet = PressurePacket(value=0.0, path_history=[wall1_id, wall2_id], birth_sample=0, delay=delay_samples)
                        # self.node_to_node[wall1_id][wall2_id] = deque([initial_packet] * delay_samples, maxlen=delay_samples)
                        self.node_to_node[wall1_id][wall2_id] = deque([0.0] * delay_samples, maxlen=delay_samples)
                    else:  # NO LOG CASE - REF
                        self.node_to_node[wall1_id][wall2_id] = deque([0.0] * delay_samples, maxlen=delay_samples)

                    if self.enable_path_logging:
                        setattr(self, f'{wall1_id}_to_{wall2_id}_delay', delay_samples)

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

        # Switch to diffuse scattering at n=3000
        # if self.n == 600 and self.specular_scattering:
        #     print(f"n={self.n}: Switching to diffuse scattering")
        #     self.specular_scattering = False
        #     N = self.num_nodes
        #     size = N - 1
        #     self.scattering_matrix = (2 / size) * np.ones((size, size)) - np.eye(size)

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
            source_pressure = input_sample * self.source_to_node_gains[src_key]

            if self.enable_path_logging == False: # NO LOG CASE - REF
                self.source_to_nodes[src_key].append(source_pressure)

            ###############################################
            else: # LOG CASE
                if source_pressure == 0:  # no source input, Just append zero and move the delay line forward
                    assert source_pressure == 0.0, f"source_pressure: {source_pressure}, should be zero!"
                    self.source_to_nodes[src_key].append(0.0)

                else:  # nonzero source input, append the packet
                    assert source_pressure != 0.0, f"source_pressure: {source_pressure}, should be nonzero!"
                    packet = PressurePacket(
                        value=source_pressure,
                        path_history=['src', wall_id],
                        birth_sample=n,
                        delay=getattr(self, f'src_to_{wall_id}_delay')
                    )
                    self.source_to_nodes[src_key].append(packet)
                    self.path_logger.log_packet(packet)
            ###############################################

        # Step 2: Process each node
        for wall_id in self.room.walls:
            src_key = f"src_to_{wall_id}"

            # Step 2a: Read arriving source pressure from source_to_node delay line
            if self.enable_path_logging==False: # NO LOG CASE - REF
                source_pressure = self.source_to_nodes[src_key][0]

            ###############################################
            else: # LOG CASE
                if isinstance(self.source_to_nodes[src_key][0], PressurePacket):
                    source_pressure = self.source_to_nodes[src_key][0].value
                else:  # there is no active source input, Just append zero and move the delay line forward
                    source_pressure = self.source_to_nodes[src_key][0]
                    assert source_pressure == 0.0, f"source_pressure: {source_pressure}, should be zero!"

            ###############################################

            # Collect incoming wave variables
            incoming_waves = []
            other_nodes = []
            # incoming_packets = []  # Store packets for path tracking

            psk = self.source_pressure_injection_coeff * source_pressure  # source pressure arriving at any (all) nodes (mostly zero except impulse sample)
            if psk != 0.0 or psk != 0:
                print("psk:", psk)

            iter = 0
            inject_ind = np.random.randint(5)

            target = get_best_reflection_target(wall_id, self.room.angle_mappings)
            for other_id in self.room.walls:
                # print("wall_id:", wall_id, "other_id:", other_id)
                if other_id != wall_id:
                    other_nodes.append(other_id)
                    # Read from delay line and add half of source pressure

                    if self.enable_path_logging == False:  # NO LOG CASE - REF
                        pki_pressure = self.node_to_node[other_id][wall_id][0]  # incoming wave from other node at n

                    ###############################################
                    else:  # LOG CASE
                        pki = self.node_to_node[other_id][wall_id][0]

                        # print("step2 n, other_id", n, other_id)
                        if isinstance(pki, PressurePacket):
                            # print("pki paketliymiş:", pki.value)
                            pki_pressure = pki.value
                            # p_tilde = pki_pressure + psk #bunu da paketlemek lazım

                            # Extend the path with current node
                            # new_packet = pki.extend_path(wall_id)
                            # incoming_packets.append(new_packet)

                        else:
                            pki_pressure = pki
                            assert pki_pressure == 0.0, f"pki_pressure: {pki_pressure}, should be zero if no incoming packets!"
                    ###############################################

                    p_tilde = pki_pressure + psk  # if p_tilde is zero, no pressure at the node

                    if self.specular_source_injection:
                        # c = 4
                        c = self.source_weighting
                        # if iter == inject_ind: # x1
                        if other_id == target:
                            incoming_waves.append(pki_pressure + c*psk)
                        else:
                            incoming_waves.append(pki_pressure + (5 - c) / 4 * psk)
                            # print("p_tilde:", p_tilde)
                    else:
                        incoming_waves.append(p_tilde)

                    iter += 1

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

                        # Create new packet for scattered wave
                        scattered_packet = PressurePacket(value=val,
                            # path_history=incoming_packets[idx].path_history + [other_id],
                            path_history="",
                            birth_sample=n,
                            # delay=incoming_packets[idx].delay + getattr(self, f'{wall_id}_to_{other_id}_delay'),
                            delay="")

                        self.outgoing_waves[wall_id][other_id] = scattered_packet

                    else:
                        val = 0.0
                        scattered_packet = PressurePacket(
                            value=val,
                            path_history="",
                            birth_sample=n,
                            delay="")

                        self.outgoing_waves[wall_id][other_id] = scattered_packet
                        assert all_zero, f"outgoingler 0 olmalıydı"  # there should be no pressure at the node
                ###############################################

            # Calculate node pressure : p_k(n) = p_Sk(n) + 2/(N-1) * Σ p_ki^+(n)
            # node_pressure = (2 / (self.num_nodes - 1)) * sum(incoming_waves)  # kullanılmıyor?
            # self.node_pressures[wall_id] = node_pressure # kullanılmıyor ?

            # Send to microphone (using outgoing waves)
            mic_key = f"{wall_id}_to_mic"
            # print("n:", n, "single nonzero=", self.instant_outgoing_waves)

            # coef = 2/5 originally
            mic_pressure = self.coef * sum(self.instant_outgoing_waves) * \
                           self.node_to_mic_gains[mic_key] * \
                           self.room.wallAttenuation[self.room.walls[wall_id].wall_index]

            if self.enable_path_logging == False: # NO LOG CASE - REF
                self.node_to_mic[mic_key].append(mic_pressure)

            ###############################################
            else: # LOG CASE
                # Create packet for microphone output
                mic_packet = PressurePacket(
                    value=mic_pressure,
                    path_history=[wall_id, 'mic'],
                    birth_sample=n,
                    delay=getattr(self, f'{wall_id}_to_mic_delay')
                )
                self.node_to_mic[mic_key].append(mic_packet if abs(mic_pressure) > 1e-10 else 0.0)
                if abs(mic_pressure) > 1e-10:
                    self.path_logger.log_packet(mic_packet)
            ###############################################
            output_sample += self.node_to_mic[mic_key][0] if not isinstance(self.node_to_mic[mic_key][0],
                                                                            PressurePacket) else self.node_to_mic[mic_key][0].value

        # Step 3: Update node-to-node connections using stored outgoing waves
        for wall_id in self.room.walls:
            for other_id in self.room.walls:
                if wall_id != other_id:
                    # Get outgoing wave and apply both walls' attenuations
                    outgoing_wave = self.outgoing_waves[wall_id][other_id]
                    sending_wall_atten = self.room.wallAttenuation[self.room.walls[wall_id].wall_index]

                    if self.enable_path_logging == False: # NO LOG CASE - REF
                        attenuated_wave = outgoing_wave * sending_wall_atten
                        self.node_to_node[wall_id][other_id].append(attenuated_wave)

                    ###############################################
                    else: # LOG CASE
                        if isinstance(outgoing_wave, PressurePacket):
                            attenuated_packet = PressurePacket(
                                value=outgoing_wave.value * sending_wall_atten,
                                path_history=outgoing_wave.path_history,
                                birth_sample=outgoing_wave.birth_sample,
                                delay=outgoing_wave.delay,
                            )
                            self.node_to_node[wall_id][other_id].append(attenuated_packet)
                    ###############################################

        # self.current_sample += 1
        return output_sample

    def _calculate_attenuations(self):
        """Calculate attenuation factors for all connections."""
        G = self.c / self.Fs  # unit distance

        # Direct sound attenuation (1/r law)
        src_mic_distance = self.room.srcPos.getDistance(self.room.micPos)
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
        print(f"Calculating RIR for {self.label}")
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