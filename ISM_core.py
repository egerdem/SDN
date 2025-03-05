from collections import deque
import numpy as np
from typing import Dict, List, Optional
from geometry import Room, Point

class ISMNetwork:
    """Core ISM implementation focusing on sample-based processing with delay lines."""

    def __init__(self, room: Room, Fs: int = 44100, c: float = 343.0):
        """Initialize ISM network.
        
        Args:
            room: Room geometry and parameters
            Fs: Sampling frequency (default: 44100)
            c: Speed of sound (default: 343.0)
        """
        self.room = room
        self.Fs = Fs
        self.c = c

        # Initialize delay lines with public access using descriptive keys
        self.source_to_mic = {}  # Direct source to microphone
        self.path_delays = {}    # Format: "path_{order}_{index}"
        self.path_gains = {}     # Format: "path_{order}_{index}"
        self._setup_delay_lines()

    def _setup_delay_lines(self):
        """Initialize all delay lines in the network."""
        # Direct source to microphone
        src_mic_distance = self.room.source.srcPos.getDistance(self.room.micPos)
        self.direct_sound_delay = int(np.floor((self.Fs * src_mic_distance) / self.c))
        self.source_to_mic["src_to_mic"] = deque([0.0] * self.direct_sound_delay, maxlen=self.direct_sound_delay)
        
        # Direct sound gain (1/r law)
        # self.direct_sound_gain = self.c / (self.Fs * src_mic_distance)
        self.direct_sound_gain = 1 / src_mic_distance

    def add_reflection_path(self, path_key: str, distance: float, reflection_order: int):
        """Add a reflection path to the network.
        
        Args:
            path_key: Unique identifier for the path
            distance: Total path length in meters
            reflection_order: Number of reflections in the path
        """
        # Calculate delay in samples
        delay_samples = int(np.floor((self.Fs * distance) / self.c))
        self.path_delays[path_key] = deque([0.0] * delay_samples, maxlen=delay_samples)
        
        # Calculate gain including wall reflections
        reflection_gain = self.room.wallAttenuation[0] ** reflection_order  # Assuming uniform wall properties
        # reflection_gain = 1
        # distance_gain = self.c / (self.Fs * distance)  # 1/r law
        distance_gain = 1 / distance  # 1/r law
        self.path_gains[path_key] = reflection_gain * distance_gain

    def process_sample(self, input_sample: float) -> float:
        """Process one sample through the network and return the output sample.
        
        Args:
            input_sample: Input sample value
            
        Returns:
            Output sample value
        """
        output_sample = 0.0

        # Process direct sound
        if input_sample != 0:
            direct_sound = input_sample * self.direct_sound_gain
            self.source_to_mic["src_to_mic"].append(direct_sound)
        else:
            self.source_to_mic["src_to_mic"].append(0.0)
        
        output_sample += self.source_to_mic["src_to_mic"][0]

        # Process reflection paths
        for path_key in self.path_delays.keys():
            if input_sample != 0:
                path_pressure = input_sample * self.path_gains[path_key]
                self.path_delays[path_key].append(path_pressure)
            else:
                self.path_delays[path_key].append(0.0)
            
            output_sample += self.path_delays[path_key][0]

        return output_sample

    def calculate_rir(self, duration: float) -> np.ndarray:
        """Calculate room impulse response.
        
        Args:
            duration: Duration of the RIR in seconds
            
        Returns:
            Room impulse response as numpy array
        """
        num_samples = int(self.Fs * duration)
        rir = np.zeros(num_samples)

        for n in range(num_samples):
            rir[n] = self.process_sample(self.room.source.signal[n])

        return rir 