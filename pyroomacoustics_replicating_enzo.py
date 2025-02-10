import random
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
from src.utils import EchoDensity as ned

"""
room_dim = [3.2, 4.0, 2.7]

room = pra.ShoeBox(room_dim, fs=44100, materials=pra.Material(0.1), max_order=50)

room.add_source([2.0, 3.5, 2.0])
room.add_microphone([2.0, 1.5, 2.0])

room.compute_rir()

wavfile.write('./Audio/pyroomacoustics_replicating_enzo.wav', 44100, np.array(room.rir))
"""

room_dim = [3.2, 4.0, 2.7]
rirs = []

for i in range(10):
    room = pra.ShoeBox(room_dim, fs=44100, materials=pra.Material(0.1), max_order=50)

    room.add_source([random.uniform(0, room_dim[0]),
                     random.uniform(0, room_dim[1]),
                     random.uniform(0, room_dim[2])])
    room.add_microphone([random.uniform(0, room_dim[0]),
                         random.uniform(0, room_dim[1]),
                         random.uniform(0, room_dim[2])])

    room.compute_rir()
    rirs.append(room.rir[0][0])

"""
plt.clf()
room.plot_rir()
plt.xscale('log')
plt.xlim(left=0.001)

plt.show()
"""
#wavfile.write('./Audio/pyroomacoustics_replicating_enzo.wav', 44100, np.array(room.rir))

echo_densities = []
shortest = 9999999999999
for rir in rirs:
    echo_density = ned.echoDensityProfile(rir)
    shortest = min(shortest, len(echo_density))
    echo_densities.append(echo_density)

for i in range(10):
    echo_densities[i] = echo_densities[i][:shortest-1]

echo_density = np.mean(echo_densities, axis=0)

plt.clf()
plt.plot(echo_density)
plt.xlabel('Time (samples)')
plt.ylabel('Normalized Echo Density')
plt.xscale('log')
plt.xlim(left=100)

plt.show()