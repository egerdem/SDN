# Your first line of Python codeimport numpy as np
import matplotlib.pyplot as plt
import numpy as np
# Constants
c = 343  # Speed of sound in air (m/s)
f = 1000  # Frequency of the sound (Hz)
omega = 2 * np.pi * f  # Angular frequency (rad/s)

# Define talker and microphone locations
talker_location = np.array([2, 2, 2])  # X (m)
mic_location = np.array([1, 1, 1])  # X' (m)

# Calculate distance R
R = np.linalg.norm(mic_location - talker_location)

t = np.linspace(0.1, 10, 500)  # Avoid division by zero with minimum distance 0.1 m

# Calculate pressure
p = np.exp(1j * omega * (R / c - t)) / (4 * np.pi * R)
p = np.abs(p)
# Plotting the pressure magnitude as a function of distance for interpretation

plt.figure(figsize=(10, 6))
plt.plot(t, p)
plt.title("Sound Pressure Magnitude vs. time")
plt.xlabel("Distance (m)")
plt.ylabel("Pressure Magnitude")
plt.grid(True)
plt.show()
