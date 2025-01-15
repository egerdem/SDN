"""Path logging mechanism for SDN implementation analysis."""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

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
    
    @property
    def reaches_mic(self) -> bool:
        """Check if this path reaches the microphone."""
        return len(self.path_history) >= 2 and self.path_history[-1] == 'mic'
    
    @property
    def order(self) -> int:
        """Get reflection order (number of bounces excluding src and mic)."""
        return len(self.path_history) - 2 if self.reaches_mic else len(self.path_history) - 1
    
    def __str__(self):
        """String representation showing path and value."""
        path_str = " -> ".join(self.path_history)
        arrival_time = self.birth_sample + self.delay if self.reaches_mic else None
        arrival_str = f", arrives at n={arrival_time}" if arrival_time is not None else ""
        return f"[n={self.birth_sample}] {path_str}: value={self.value:.6f}, delay={self.delay}{arrival_str}"


class PathLogger:
    """Logs and analyzes pressure propagation paths in SDN."""
    
    def __init__(self, threshold: float = 1e-20):
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
            matching_packets = [p for p in packets if p.birth_sample + p.delay == sample_idx]
            if matching_packets:
                active_paths[path_key] = matching_packets[-1]  # Get most recent packet
        return active_paths
    
    def get_complete_paths_sorted(self) -> List[Tuple[str, PressurePacket]]:
        """Get all paths that reach the mic, sorted by arrival time."""
        complete_paths = []
        for path_key, packets in self.paths.items():
            mic_packets = [p for p in packets if p.reaches_mic]
            complete_paths.extend((path_key, p) for p in mic_packets)
        
        # Sort by arrival time (birth_sample + delay)
        return sorted(complete_paths, key=lambda x: x[1].birth_sample + x[1].delay)
    
    def get_paths_by_order(self, order: int) -> List[Tuple[str, PressurePacket]]:
        """Get all paths of a specific reflection order that reach the mic."""
        order_paths = []
        for path_key, packets in self.paths.items():
            matching_packets = [p for p in packets if p.reaches_mic and p.order == order]
            order_paths.extend((path_key, p) for p in matching_packets)
        return order_paths
    
    def print_path_summary(self, path_key: Optional[str] = None):
        """Print summary of paths, optionally filtered by path_key."""
        paths_to_print = {path_key: self.paths[path_key]} if path_key else self.paths
        
        for key, packets in paths_to_print.items():
            print(f"\nPath: {key}")
            print("-" * 50)
            for packet in packets:
                print(packet)
            print("-" * 50)
    
    def print_complete_paths_summary(self):
        """Print summary of all paths that reach the mic, grouped by order."""
        complete_paths = self.get_complete_paths_sorted()
        if not complete_paths:
            print("No complete paths found.")
            return
            
        current_order = None
        for path_key, packet in complete_paths:
            if packet.order != current_order:
                current_order = packet.order
                print(f"\nOrder {current_order} reflections:")
                print("-" * 50)
            print(f"{path_key}: arrives at n={packet.birth_sample + packet.delay}, value={packet.value:.6f}")

def deque_plotter(a, label):
    val = []
    for item in a:
        # Check if the item is a PressurePacket
        if hasattr(item, 'value'):
            val.append(item.value)
        else:
            val.append(item)

    # Create a scatter plot
    plt.figure()
    plt.scatter(range(len(val)), val)
    plt.title(label)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    return