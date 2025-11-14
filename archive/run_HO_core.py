# This script runs the HO-SDN centroid simulation extracted from the WASPAA submission:
# LOW-COMPLEXITY HIGHER ORDER SCATTERING DELAY NETWORKS

#%% 1-Import Libraries
import scipy as sp
import os
import timeit
import pickle
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pickle
import pyroomacoustics as pra
import scipy.io as sio
import warnings
import importlib
import sys
from pprint import pprint
from copy import deepcopy

# Add the SDNPy directory to the Python path
sys.path.append('SDN-Simplest_Hybrid_HO-SDN/SDNPy')

# Now import from the correct path
from src import Geometry as geom
from src import Signal as sig
from src import Source as src
from src import Microphone as mic
from src import Simulation as sim
from src import Simulation_HO_SDN_centroid as sim_HO_SDN_centroid
from src.utils import EchoDensity as ned
from src.utils.rrdecay import EDC
from src.utils.rrdecay import EAC

#%% 3-Initiate model parameters
exp_path='Results/waspaa_att02/'
frameSize = None  # None frame size means it adapts to the delay lines
outputs = []

Fs = 44100
figure_size = 1.5

sdn_parameters_list = [{'order': 1,
                        'connection': 'full',
                        'node selection': 'all',
                        'matrix': 'isotropic',
                        'skeleton extras': 0},
                        {'order': 2,
                        'connection': 'full',
                        'node selection': 'all',
                        'matrix': 'isotropic',
                        'skeleton extras': 0},
                        {'order': 3,
                        'connection': 'full',
                        'node selection': 'all',
                        'matrix': 'isotropic',
                        'skeleton extras': 0},
                       ]

pprint(sdn_parameters_list)

room_parameters_list = [      
                        {'subfolder name': 'medium room 2 ',    #<----the room used in the waspaa submission
                         'width': 6.0, 'height': 4.0, 'depth': 7.0,
                         'mic x': 1.2, 'mic y': 2.4, 'mic z': 1.8,
                        'source x': 3.6, 'source y': 1.3, 'source z' : 5.3,
                         'l.o.s. distance': 4.373,
                         'absorption': 0.1, 
                         'air': None,
                         'duration': 2.0,
                         'duration_inSec': 2.0,
                         'estimated T60': 1.5,
                         },   
                        ]

pprint(room_parameters_list)
for room_param in room_parameters_list:
    if (room_param['source x']>room_param['width'] or
        room_param['source y']>room_param['height'] or
        room_param['source z']>room_param['depth'] or
        room_param['mic x']>room_param['width'] or
        room_param['mic y']>room_param['height'] or
        room_param['mic z']>room_param['depth']
        ):
        warnings.warn('WARNING: Source or mic position out of the room')

for room_index, room_parameters in enumerate(room_parameters_list):
    print('Starting room', room_index+1, 'out of', len(room_parameters_list))

    room_parameters['reflection'] = -1 * np.sqrt(1 - room_parameters['absorption'])
    
    room_parameters['duration'] = int(room_parameters['duration'] * Fs)

    if 'l.o.s. distance' in room_parameters.keys():
        room_parameters['l.o.s. delay'] = round(Fs * (room_parameters['l.o.s. distance'] / 343))
    else:
        room_parameters['l.o.s. delay'] = 0

    rir_start = max(0, room_parameters['l.o.s. delay'] - 10)
    rir_end = rir_start + int(Fs / 40)

    data = np.zeros(room_parameters['duration'], dtype=float)
    data[0] = 1.0
    signal = sig.Signal(Fs, data)

    # make the room and choose its shape
    room = geom.Room()
    room.shape = geom.Cuboid(room_parameters['width'],
                             room_parameters['height'],
                             room_parameters['depth'])
    nWalls = room.shape.nWalls

    filt_b = {}
    filt_a = {}

    room.wallFilters = [None] * 6
    room.wallAttenuation = [room_parameters['reflection']] * 6

    # choose position of source
    srcPosition = geom.Point(room_parameters['source x'],
                             room_parameters['source y'],
                             room_parameters['source z'])
    # create source at specified position, with input signal
    source = src.Source(srcPosition, signal)

    #choose position of mic
    micPosition = geom.Point(room_parameters['mic x'],
                             room_parameters['mic y'],
                             room_parameters['mic z'])
    # create microphone at specified position
    microphone = mic.Microphone(micPosition)

#%%                     9-RUN HO-SDN Centroid
importlib.reload(sim_HO_SDN_centroid)
for room_index, room_parameters in enumerate(room_parameters_list):
    for sdn_index, sdn_parameters in enumerate(sdn_parameters_list):
        print('\tStarting SDN order', sdn_parameters['order'],
              'connection', sdn_parameters['connection'],
              'matrix', sdn_parameters['matrix'],
              'skeleton extras', sdn_parameters['skeleton extras'],
              '( SDN', sdn_index+1, 'out of', len(sdn_parameters_list), ')')


        simulate_mod = sim_HO_SDN_centroid.Simulation(deepcopy(room),
                                    source, microphone,
                                    frameSize, room_parameters['duration'],
                                    sdn_parameters['order'],
                                    sdn_parameters['connection'],
                                    room_parameters['air'],
                                    sdn_parameters['skeleton extras'],
                                    sdn_parameters['matrix'])
        
        
        
        # start the timer
        start = timeit.default_timer()

        # Modifyed HO-SDN
        audio, multiAudio, sdnKickIn = simulate_mod.run()
        
        multiAudio = multiAudio / np.max(np.abs(audio))
        audio = audio / np.max(np.abs(audio))
        sdnKickIn = sdnKickIn/ np.max(np.abs(sdnKickIn))
        # stop the timer
        stop = timeit.default_timer()

        print('\tElapsed time:', stop - start)

        outputs.append({})
        outputs[-1]['audio'] = audio
        outputs[-1]['time'] = stop - start
        outputs[-1]['lines'] = simulate_mod.nProplines
        outputs[-1]['nodes'] = len(simulate_mod.all_nodes)
        outputs[-1]['label'] = 'HO-SDN centroid N='+str(sdn_parameters['order'])
        outputs[-1]['linewidth'] = 1 
        outputs[-1]['linestyle'] = '--'
        outputs[-1]['sdnKickIn'] = sdnKickIn 

#%% Plot RIR
plt.figure(figsize=(6.4*figure_size, 4.8*figure_size))
for output in outputs:
    plt.plot(np.array(range(len(output['audio'])))/Fs,
             output['audio'],
             label=output['label'],
             linewidth=output['linewidth'],
             linestyle=output['linestyle'])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Plot early part of RIR
plt.figure(figsize=(6.4*figure_size, 4.8*figure_size))
for output in outputs:
    plt.plot(np.array(range(0, rir_end*2))/Fs,
             output['audio'][0:rir_end*2],
             label=output['label'],
             linewidth=output['linewidth'],
             linestyle=output['linestyle'])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Early RIR')
plt.legend()
plt.grid(True)
plt.show() 