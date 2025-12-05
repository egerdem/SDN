import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import geometry
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')

Fs = 44100
# Define the Room
room_parameters = {'width': 9, 'depth': 7, 'height': 4,
                   'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   'absorption': 0.2, 'duration': 0.2
                   }

# Setup room
room = geometry.Room(room_parameters['width'],
                     room_parameters['depth'],
                     room_parameters['height'])

room_dim = np.array([room_parameters['width'],
                         room_parameters['depth'],
                         room_parameters['height']])

source_loc = np.array([room_parameters['source x'],
                       room_parameters['source y'],
                       room_parameters['source z']])

mic_loc = np.array([room_parameters['mic x'],
                    room_parameters['mic y'],
                    room_parameters['mic z']])


# pra_room = pra.ShoeBox(room_dim, fs=Fs,
#                                materials=pra.Material(room_parameters['absorption']),
#                                max_order=6,
#                                air_absorption=False)

pra_room = pra.ShoeBox(room_dim, fs=Fs,
                               materials=pra.Material(room_parameters['absorption']),
                               max_order=6,
                               air_absorption=False)

pra_room.set_sound_speed(343)
pra_room.add_source(source_loc).add_microphone(mic_loc)

pra_room.compute_rir()
pra_mod = pra_room.rir[0][0][:int(room_parameters['duration']*Fs)]
pra_mod = pra_mod / np.max(np.abs(pra_mod))
pra_room.plot_rir()
plt.show()

plt.figure()
plt.plot(pra_mod)
plt.title('ISM PRA Trials')
plt.show()