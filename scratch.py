import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import geometry
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5


# Parameters
Fs = 44100  # Sample rate
c = 343.0   # Speed of sound
duration = 0.1  # seconds
num_samples = int(Fs * duration)
num_paths = 10

# Generate random distances (between 1 and 10 meters)
np.random.seed(4)  # for reproducibility
distances = np.random.uniform(1, 10, num_paths)
delay_samples_list = [int(np.floor((Fs * d) / c)) for d in distances]

# Create sorted indices based on delay samples
sorted_indices = np.argsort(delay_samples_list)
distances = distances[sorted_indices]
delay_samples_list = [delay_samples_list[i] for i in sorted_indices]

print("\nPath details (ordered by arrival time):")
for i, (dist, delay) in enumerate(zip(distances, delay_samples_list)):
    print(f"Path {i}: {dist:.2f}m -> {delay} samples")

# Calculate attenuations
attenuations = [1.0 / dist for dist in distances]

# Setup the figure for animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
plt.tight_layout(pad=3.0)

# Initialize lines for animation
line1, = ax1.plot([], [], 'b-')
stems = []
scatter_points = []

# Setup axis limits and labels
ax1.set_xlim(0, num_samples)
ax1.set_ylim(min(attenuations) - 0.1, max(attenuations) + 0.1)
ax1.set_title('Building Up Room Impulse Response')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

ax2.set_xlim(0, num_samples)
ax2.set_ylim(min(attenuations) - 0.1, max(attenuations) + 0.1)
ax2.set_title('Impulse Arrival Times and Amplitudes')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Amplitude (1/distance)')
ax2.grid(True)

# Function to calculate partial output up to n paths
def calculate_partial_output(n_paths):
    output = np.zeros(num_samples)
    delay_lines = []
    
    # Create delay lines for paths up to n_paths
    for i in range(n_paths):
        delay_samples = delay_samples_list[i]
        delay_line = deque([0.0] * delay_samples, maxlen=delay_samples)
        delay_lines.append(delay_line)
    
    # Process samples
    for n in range(num_samples):
        input_sample = 1.0 if n == 0 else 0.0
        for i, delay_line in enumerate(delay_lines):
            delay_line.append(input_sample * attenuations[i])
            output[n] += delay_line[0]
    
    return output

# Animation initialization function
def init():
    line1.set_data([], [])
    return (line1,)

# Animation update function
def update(frame):
    # Calculate output for current number of paths
    n_paths = frame + 1
    output = calculate_partial_output(n_paths)
    
    # Update IR plot
    line1.set_data(range(num_samples), output)
    
    # Update stem plot
    ax2.clear()
    ax2.set_title('Impulse Arrival Times and Amplitudes')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Amplitude (1/distance)')
    ax2.grid(True)
    ax2.set_xlim(0, num_samples)
    ax2.set_ylim(min(attenuations) - 0.1, max(attenuations) + 0.1)
    ax2.stem(delay_samples_list[:n_paths], attenuations[:n_paths], basefmt=' ')
    
    # Update titles to show progress
    ax1.set_title(f'Building Up Room Impulse Response ({n_paths} of {num_paths} paths)')
    
    return (line1,)

# Create animation
anim = FuncAnimation(fig, update, frames=num_paths, init_func=init,
                    interval=1000, blit=False)  # 1000ms = 1s between frames

plt.show()
