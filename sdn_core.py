from collections import deque
import numpy as np
from typing import Dict, List, Optional
from geometry import Room, Point
from path_logger import PathLogger

class DelayNetwork:
    """Core SDN implementation focusing on sample-based processing with accessible delay lines."""
    
    def __init__(self, room: Room, Fs: int = 44100, c: float = 343.0, source_pressure_injection_coeff: float = 0.5,
                 use_identity_scattering: bool = False,
                 ignore_wall_absorption: bool = False,
                 ignore_src_node_atten: bool = False,
                 ignore_node_mic_atten: bool = False,
                 enable_path_logging: bool = False):
        """Initialize SDN with test flags.
        
        Args:
            room: Room geometry and parameters
            Fs: Sampling frequency (default: 44100)
            c: Speed of sound (default: 343.0)
            use_identity_scattering: If True, use identity matrix for scattering (default: False)
            ignore_wall_absorption: If True, set wall reflection coefficients to 1 (default: False)
            ignore_src_node_atten: If True, set source-to-node gains to 1 (default: False)
            ignore_node_mic_atten: If True, set node-to-mic gains to 1 (default: False)
            enable_path_logging: If True, enable detailed path logging (default: False)
        """
        self.room = room
        self.Fs = Fs
        self.c = c
        self.source_pressure_injection_coeff = source_pressure_injection_coeff

        # Test flags
        self.use_identity_scattering = use_identity_scattering
        self.ignore_wall_absorption = ignore_wall_absorption
        self.ignore_src_node_atten = ignore_src_node_atten
        self.ignore_node_mic_atten = ignore_node_mic_atten
        
        # Initialize path logger if enabled
        self.enable_path_logging = enable_path_logging
        self.path_logger = PathLogger() if enable_path_logging else None
        
        # Network parameters
        self.num_nodes = len(room.walls)  # Number of scattering nodes (6 for cuboid)
        self.scattering_matrix = self._create_scattering_matrix()
        
        # Initialize delay lines with public access using descriptive keys
        self.source_to_mic = {}  # Direct source to microphone
        self.source_to_nodes = {}    # Format: "src_to_{wall_id}"
        self.node_to_mic = {}        # Format: "{wall_id}_to_mic"
        self.node_to_node = {}       # Format: "{wall1_id}_to_{wall2_id}"
        self._setup_delay_lines()
        
        # Initialize attenuation factors with matching keys
        # self.source_to_mic_gain is added inside the _calculate_attenuations() function
        self.source_to_node_gains = {}  # g_sk, Format: "src_to_{wall_id}"
        self.node_to_mic_gains = {}     # g_km, Format: "{wall_id}_to_mic"
        self._calculate_attenuations()
        
        # State variables
        self.current_sample = 0
        self.node_pressures = {wall_id: 0.0 for wall_id in room.walls}
        
        # For storing outgoing wave variables at each node
        self.outgoing_waves = {wall_id: {} for wall_id in room.walls}
        for wall1_id in room.walls:
            for wall2_id in room.walls:
                if wall1_id != wall2_id:
                    self.outgoing_waves[wall1_id][wall2_id] = 0.0
    
    def _create_scattering_matrix(self) -> np.ndarray:
        """Create the scattering matrix S = (2/(N-1))1_{(N-1)×(N-1)} - I
        This matrix is used for each node's scattering operation.
        Size is (N-1)×(N-1) as each node has N-1 connections (excluding self)."""
        N = self.num_nodes
        size = N - 1  # Matrix size for each node's operation
        
        if self.use_identity_scattering:
            return np.eye(size)
        else:
            return (2/size) * np.ones((size, size)) - np.eye(size)

    def _setup_delay_lines(self):
        """Initialize all delay lines in the network."""

        # Direct source to microphone
        key = "src_to_mic"
        src_mic_distance = self.room.srcPos.getDistance(self.room.micPos)
        direct_sound_delay = int(np.floor((self.Fs * src_mic_distance) / self.c))
        self.source_to_mic[key] = deque([0.0] * direct_sound_delay, maxlen=direct_sound_delay)
        
        if self.enable_path_logging:
            self.direct_sound_delay = direct_sound_delay

        # Source to nodes
        for wall_id, wall in self.room.walls.items():
            key = f"src_to_{wall_id}"
            dist = wall.node_positions.getDistance(self.room.source.srcPos)
            delay_samples = int(np.floor((self.Fs * dist) / self.c))
            self.source_to_nodes[key] = deque([0.0] * delay_samples, maxlen=delay_samples)
            
            if self.enable_path_logging:
                setattr(self, f'src_to_{wall_id}_delay', delay_samples)
        
        # Nodes to microphone
        for wall_id, wall in self.room.walls.items():
            key = f"{wall_id}_to_mic"
            dist = wall.node_positions.getDistance(self.room.micPos)
            delay_samples = int(np.floor((self.Fs * dist) / self.c))
            self.node_to_mic[key] = deque([0.0] * delay_samples, maxlen=delay_samples)
            
            if self.enable_path_logging:
                setattr(self, f'{wall_id}_to_mic_delay', delay_samples)
        
        # Between nodes
        for wall1_id, wall1 in self.room.walls.items():
            self.node_to_node[wall1_id] = {}
            for wall2_id, wall2 in self.room.walls.items():
                if wall1_id != wall2_id:
                    dist = wall1.node_positions.getDistance(wall2.node_positions)
                    delay_samples = int(np.floor((self.Fs * dist) / self.c))
                    self.node_to_node[wall1_id][wall2_id] = deque([0.0] * delay_samples, maxlen=delay_samples)
                    
                    if self.enable_path_logging:
                        setattr(self, f'{wall1_id}_to_{wall2_id}_delay', delay_samples)
    
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
            self.node_to_mic_gains[mic_key] = 1.0 / (1.0 + mic_dist/src_dist) if not self.ignore_node_mic_atten else 1.0

    def process_sample(self, input_sample, n) -> float:
        """Process one sample through the network and return the output sample."""
        output_sample = 0.0

        # Step 0: Source to mic direct sound
        self.source_to_mic["src_to_mic"].append(input_sample)
        direct_sound = self.source_to_mic["src_to_mic"][0] * self.source_to_mic_gain
        output_sample += direct_sound
        
        if self.enable_path_logging and abs(direct_sound) > 1e-10:
            self.path_logger.log_direct_path(n, direct_sound, self.direct_sound_delay, 
                                           self.source_to_mic_gain)

        # Step 1: Distribute input to nodes
        for wall_id in self.room.walls:
            src_key = f"src_to_{wall_id}"
            source_pressure = input_sample * self.source_to_node_gains[src_key]
            self.source_to_nodes[src_key].append(source_pressure)
            
            if self.enable_path_logging and abs(source_pressure) > 1e-10:
                delay = getattr(self, f'src_to_{wall_id}_delay')
                self.path_logger.log_source_to_node(n, wall_id, source_pressure, delay,
                                                  self.source_to_node_gains[src_key])

        # Step 2: Process each node
        for wall_id in self.room.walls:
            src_key = f"src_to_{wall_id}"
            source_pressure = self.source_to_nodes[src_key][0]  # Get current source pressure

            # Debug prints for ceiling node
            # if wall_id == 'ceiling':
            #     # Check for any non-zero incoming waves
            #     has_nonzero_input = False
            #     for other_id in self.room.walls:
            #         if other_id != wall_id:
            #             if abs(self.node_to_node[other_id][wall_id][0]) > 1e-10 or abs(source_pressure) > 1e-10:
            #                 has_nonzero_input = True
            #                 break
            #
            #     if has_nonzero_input:
            #         print(f"\n=== Ceiling Node at sample {n} ===")
            #         print(f"Direct source pressure: {source_pressure:.6f}")
            #         print("\nIncoming waves from other nodes:")
            #         for other_id in self.room.walls:
            #             if other_id != wall_id:
            #                 wave = self.node_to_node[other_id][wall_id][0]
            #                 if abs(wave) > 1e-10:  # Only print non-negligible waves
            #                     print(f"  From {other_id}: {wave:.6f}")

            # Collect incoming wave variables
            incoming_waves = []
            other_nodes = []
            for other_id in self.room.walls:
                if other_id != wall_id:
                    other_nodes.append(other_id)
                    # Read from delay line and add half of source pressure
                    pki = self.node_to_node[other_id][wall_id][0] # incoming wave from other node
                    psk = self.source_pressure_injection_coeff * source_pressure
                    p_tilde = pki + psk

                    if self.enable_path_logging and abs(pki) > 1e-10:
                        delay = getattr(self, f'{other_id}_to_{wall_id}_delay')
                        self.path_logger.log_node_to_node(n, other_id, wall_id, pki, delay,
                                                        1.0, source_pressure, pki)

                    incoming_waves.append(p_tilde)

            # Apply scattering matrix to get outgoing waves
            outgoing_waves = np.dot(self.scattering_matrix, incoming_waves)

            # Debug prints for ceiling node (continued)
            # if wall_id == 'ceiling' and has_nonzero_input:
            #     print("\nProcessed waves:")
            #     print(f"Source injection coefficient: {self.source_pressure_injection_coeff}")
            #     print("Incoming waves after source injection:")
            #     for i, other_id in enumerate(other_nodes):
            #         print(f"  To {other_id}: {incoming_waves[i]:.6f}")
            #     print("\nOutgoing waves after scattering:")
            #     for i, other_id in enumerate(other_nodes):
            #         print(f"  To {other_id}: {outgoing_waves[i]:.6f}")
            #     print(f"total incoming waves: {sum(incoming_waves):.6f}")
            #     print(f"total outgoing waves: {sum(outgoing_waves):.6f}")
            #     print("-" * 50)

            # Store outgoing waves for each connection
            for idx, other_id in enumerate(other_nodes):
                self.outgoing_waves[wall_id][other_id] = outgoing_waves[idx]

            # Calculate node pressure: p_k(n) = p_Sk(n) + 2/(N-1) * Σ p_ki^+(n)
            node_pressure =  (2/(self.num_nodes-1)) * sum(incoming_waves) # I removed the src contribution (as already done above?)
            # node_pressure =  1 * sum(incoming_waves) # I removed the src contribution (as already done above?)
            self.node_pressures[wall_id] = node_pressure

            # Send to microphone (using outgoing waves)
            mic_key = f"{wall_id}_to_mic"
            # mic_pressure = (2/(self.num_nodes-1)) * sum(outgoing_waves) * self.node_to_mic_gains[mic_key]
            mic_pressure = 1 * sum(outgoing_waves) * self.node_to_mic_gains[mic_key]
            self.node_to_mic[mic_key].append(mic_pressure)
            output_sample += self.node_to_mic[mic_key][0]
            
            if self.enable_path_logging and abs(mic_pressure) > 1e-10:
                delay = getattr(self, f'{wall_id}_to_mic_delay')
                self.path_logger.log_node_to_mic(n, wall_id, mic_pressure, delay,
                                               self.node_to_mic_gains[mic_key],
                                               source_pressure)

        # Step 3: Update node-to-node connections using stored outgoing waves
        for wall_id in self.room.walls:
            for other_id in self.room.walls:
                if wall_id != other_id:
                    # Get outgoing wave and apply both walls' attenuations
                    outgoing_wave = self.outgoing_waves[wall_id][other_id]
                    if self.ignore_wall_absorption:
                        sending_wall_atten = receiving_wall_atten = 1.0
                    else:
                        sending_wall_atten = self.room.wallAttenuation[self.room.walls[wall_id].wall_index]
                        receiving_wall_atten = self.room.wallAttenuation[self.room.walls[other_id].wall_index]
                        # receiving_wall_atten = 1

                    attenuated_wave = outgoing_wave * sending_wall_atten * receiving_wall_atten
                    self.node_to_node[wall_id][other_id].append(attenuated_wave)
        
        self.current_sample += 1
        return output_sample
    
    def calculate_rir(self, duration):
        """Calculate room impulse response.
        
        Args:
            duration: Duration of the RIR in seconds
            
        Returns:
            Room impulse response as numpy array
        """
        num_samples = int(self.Fs * duration)
        rir = np.zeros(num_samples)
        
        # Step 1: Input the impulse (initial energy)
        rir[0] = self.process_sample(1.0, n=0)
        
        # Step 2: Propagate the energy through the network
        for n in range(1, num_samples):
            # No new energy input (zero), just propagate existing energy
            propagated_energy = self.propagate_network(n)
            rir[n] = propagated_energy
        
        return rir

    def propagate_network(self, n):
        """Propagate existing energy through the network without new input.
        
        Returns:
            Output sample at microphone
        """
        return self.process_sample(0, n)  # Using existing process_sample with zero input
    
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
    