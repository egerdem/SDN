"""Path logging mechanism for SDN implementation analysis."""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class PressurePacket:
    """Represents a pressure value with its propagation history."""
    value: float
    path_history: List[str]  # List of nodes visited (e.g., ['src', 'ceiling', 'mic'])
    birth_sample: int        # Sample index when this packet was created
    delay: int              # Accumulated delay in samples
    
    def extend_path(self, node: str, additional_delay: int = 0) -> 'PressurePacket':
        """Create new packet with extended path."""
        return PressurePacket(
            value=self.value,
            path_history=self.path_history + [node],
            birth_sample=self.birth_sample,
            delay=self.delay + additional_delay
        )
    
    def __str__(self):
        """String representation showing path and value."""
        path_str = " -> ".join(self.path_history)
        return f"[n={self.birth_sample}] {path_str}: value={self.value:.6f}, delay={self.delay}"

class PathLogger:
    """Logs and analyzes pressure propagation paths in SDN."""
    
    def __init__(self, threshold: float = 1e-10):
        self.threshold = threshold  # Minimum pressure to log
        self.paths: Dict[str, List[PressurePacket]] = {}
        
    def log_packet(self, packet: PressurePacket):
        """Log a pressure packet if its value is above threshold."""
        if abs(packet.value) > self.threshold:
            # Create path key from history
            path_key = "->".join(packet.path_history)
            
            # Initialize list for this path if needed
            if path_key not in self.paths:
                self.paths[path_key] = []
            
            # Add packet to path history
            self.paths[path_key].append(packet)
    
    def get_paths_through_node(self, node_id: str) -> Dict[str, List[PressurePacket]]:
        """Get all paths that pass through a specific node."""
        return {k: v for k, v in self.paths.items() if node_id in k}
    
    def get_active_paths_at_sample(self, sample_idx: int) -> Dict[str, PressurePacket]:
        """Get all paths active at a specific sample index."""
        active_paths = {}
        for path_key, packets in self.paths.items():
            matching_packets = [p for p in packets if p.birth_sample == sample_idx]
            if matching_packets:
                active_paths[path_key] = matching_packets[-1]  # Get most recent packet
        return active_paths
    
    def print_path_summary(self, path_key: Optional[str] = None):
        """Print summary of paths, optionally filtered by path_key."""
        paths_to_print = {path_key: self.paths[path_key]} if path_key else self.paths
        
        for key, packets in paths_to_print.items():
            print(f"\nPath: {key}")
            print("-" * 50)
            for packet in packets:
                print(packet)
            print("-" * 50) 