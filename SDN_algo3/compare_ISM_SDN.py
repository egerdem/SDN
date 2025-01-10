from src import Geometry as geom
from src import Signal as sig
from src import Source as src
from src import Microphone as mic
from src import Simulation as sim
from src.utils import EchoDensity as ned
from src.utils.rrdecay import EDC
from src.utils.rrdecay import rrdecay

import os
import csv
import timeit
import pickle
import numpy as np
from pprint import pprint
from copy import deepcopy
from scipy.signal import butter
from scipy.signal import firls
from scipy.signal import freqz
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write

import pyroomacoustics as pra

Fs = 44100

# folder_name = 'Proper simulations'
folder_name = 'Tests/picky mistake'
exp_name = 'number'
save_stuff = False
frameSize = None  # None frame size means it adapts to the delay lines
run_pra = True

pra_order = 100
figure_size = 1.5
trim_others = True

short_ned_end = int(Fs/8)
short_ned_window = 30
long_ned_end = int(Fs*2)
long_ned_window = 100

sdn_parameters_list = [{'order': 1,
                        'connection': 'full',
                        'node selection': 'all',
                        'matrix': 'isotropic',
                        'skeleton extras': 0},
                       ]
pprint(sdn_parameters_list)

room_parameters_list = [      
                        {'subfolder name': 'medium room',
                         'width': 9, 'height': 7, 'depth': 4,
                         'source x': 4.5, 'source y': 3.5, 'source z': 2,
                         'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                         'l.o.s. distance': 4.373,
                         'absorption': 0.1,
                         'air': {'humidity': 50,
                                 'temperature': 20,
                                 'pressure': 100},
                         'duration': 2.0,
                         'estimated T60': 1.5,
                         'catt path': '../Audio/Proper simulations/medium room/CATT medium room.wav'
                         },
                        ]

pprint(room_parameters_list)

for room_index, room_parameters in enumerate(room_parameters_list):
    print('Starting room', room_index+1, 'out of', len(room_parameters_list))

    # room_parameters['absorption'] = 1 - room_parameters['reflection']**2
    room_parameters['reflection'] = 1 * np.sqrt(1 - room_parameters['absorption'])

    room_parameters['duration'] = int(room_parameters['duration'] * Fs)

    if 'l.o.s. distance' in room_parameters.keys():
        room_parameters['l.o.s. delay'] = round(Fs * (room_parameters['l.o.s. distance'] / 343))
    else:
        room_parameters['l.o.s. delay'] = 0

    rir_start = max(0, room_parameters['l.o.s. delay'] - 10)
    rir_end = rir_start + int(Fs / 20)

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

    outputs = []

    # if room_parameters['catt path'] is not None:
    #     _, catt = read(room_parameters['catt path'])
    #     catt = np.concatenate((np.zeros(room_parameters['l.o.s. delay']), catt))
    #     catt = catt / np.max(np.abs(catt))
    #     if trim_others:
    #         catt = catt[:room_parameters['duration']]
    #     outputs.append({})
    #     outputs[-1]['audio'] = catt
    #     outputs[-1]['label'] = 'CATT-acoustics'
    #     if save_stuff:
    #         if not os.path.exists("../Audio/" + folder_name +
    #                               "/" + exp_name +
    #                               "/" + room_parameters['subfolder name']):
    #             os.makedirs("../Audio/" + folder_name +
    #                         "/" + exp_name +
    #                         "/" + room_parameters['subfolder name'])
    #         write("../Audio/" + folder_name +
    #               "/" + exp_name +
    #               "/" + room_parameters['subfolder name'] +
    #               "/CATT " + room_parameters['subfolder name'] +
    #               ".wav", Fs, catt)

    if run_pra:
        print('\tStarting pyroomacoustics')

        room_dim = np.array([room_parameters['width'],
                             room_parameters['height'],
                             room_parameters['depth']])

        source_loc = np.array([room_parameters['source x'],
                               room_parameters['source y'],
                               room_parameters['source z']])
        mic_loc = np.array([room_parameters['mic x'],
                            room_parameters['mic y'],
                            room_parameters['mic z']])

        if room_parameters['air'] is not None:
            pra_room = pra.ShoeBox(room_dim, fs=Fs,
                                   materials=pra.Material(room_parameters['absorption']),
                                   max_order=pra_order,
                                   temperature=room_parameters['air']['temperature'],
                                   humidity=room_parameters['air']['humidity'], air_absorption=True)
        else:
            pra_room = pra.ShoeBox(room_dim, fs=Fs,
                                   materials=pra.Material(room_parameters['absorption']),
                                   max_order=pra_order,
                                   air_absorption=False)
        pra_room.set_sound_speed(343)
        pra_room.add_source(source_loc).add_microphone(mic_loc)

        pra_room.compute_rir()

        if trim_others:
            pra_mod = pra_room.rir[0][0][38:room_parameters['duration']]
        else:
            pra_mod = pra_room.rir[0][0][38:]
        pra_mod = pra_mod / np.max(np.abs(pra_mod))
        outputs.append({})
        outputs[-1]['audio'] = pra_mod
        outputs[-1]['label'] = 'PyRoomAcoustics'
        if save_stuff:
            if not os.path.exists("../Audio/" + folder_name +
                                  "/" + exp_name +
                                  "/" + room_parameters['subfolder name']):
                os.makedirs("../Audio/" + folder_name +
                            "/" + exp_name +
                            "/" + room_parameters['subfolder name'])
            write("../Audio/" + folder_name +
                  "/" + exp_name +
                  "/" + room_parameters['subfolder name'] +
                  "/PRA " + room_parameters['subfolder name'] +
                  ".wav", Fs, pra_mod)

        print('\tFinished pyroomacoustics')

    for sdn_index, sdn_parameters in enumerate(sdn_parameters_list):
        print('\tStarting SDN order', sdn_parameters['order'],
              'connection', sdn_parameters['connection'],
              'matrix', sdn_parameters['matrix'],
              'skeleton extras', sdn_parameters['skeleton extras'],
              '( SDN', sdn_index+1, 'out of', len(sdn_parameters_list), ')')

        wav_path = "../Audio/" + folder_name + \
                   "/" + exp_name + \
                   "/" + room_parameters['subfolder name'] + \
                   "/SDN order " + str(sdn_parameters['order']) + \
                   "/connection " + sdn_parameters['connection'] + \
                   "/matrix " + sdn_parameters['matrix'] + \
                   "/skeleton extras " + str(sdn_parameters['skeleton extras'])

        if not os.path.exists(wav_path):
            # create the simulation
            simulate = sim.Simulation(deepcopy(room),
                                      source, microphone,
                                      frameSize, room_parameters['duration'],
                                      sdn_parameters['order'],
                                      sdn_parameters['connection'],
                                      room_parameters['air'],
                                      sdn_parameters['skeleton extras'],
                                      sdn_parameters['matrix'])

            # start the timer
            start = timeit.default_timer()
            # run the simulation
            try:
                audio = simulate.run()
            except Exception as e:
                print('An error occurred during simulation.')
                print(e)
                audio = np.zeros_like(signal.data)

            audio = audio / np.max(np.abs(audio))

            # stop the timer
            stop = timeit.default_timer()

            print('\tElapsed time:', stop - start)

            outputs.append({})
            outputs[-1]['audio'] = audio
            outputs[-1]['time'] = stop - start
            outputs[-1]['lines'] = simulate.nProplines
            outputs[-1]['nodes'] = len(simulate.all_nodes)
            outputs[-1]['label'] = 'SDN Ord: ' + str(sdn_parameters['order']) +\
                                   ' - Mat: ' + sdn_parameters['matrix'] +\
                                   ' - Con: ' + sdn_parameters['connection'] +\
                                   ' - Ext: ' + str(sdn_parameters['skeleton extras'])

            if save_stuff:
                if not os.path.exists('../Data/' + folder_name + '/' + room_parameters['subfolder name']):
                    os.makedirs('../Data/' + folder_name + '/' + room_parameters['subfolder name'])
                with open('../Data/' + folder_name + '/' + room_parameters['subfolder name'] + '/outputs ' + exp_name + '.pickle',
                          'wb') as f:
                    pickle.dump(outputs, f)

                os.makedirs(wav_path)
                write(wav_path + "/audio.wav", Fs, audio)
        else:
            print('\tAlready exists.')
            _, audio = read(wav_path + "/audio.wav")
            outputs.append({})
            outputs[-1]['audio'] = audio
            outputs[-1]['label'] = 'SDN Ord: ' + str(sdn_parameters['order']) +\
                                   ' - Mat: ' + sdn_parameters['matrix'] +\
                                   ' - Con: ' + sdn_parameters['connection'] +\
                                   ' - Ext: ' + str(sdn_parameters['skeleton extras'])

   
    plt.figure(figsize=(6.4*figure_size, 4.8*figure_size))
    for output in outputs:
        plt.plot(np.array(range(len(output['audio'])))/Fs,
                 output['audio'],
                 label=output['label'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('RIR amplitude')
    plt.title('RIR')
    plt.legend()
    if save_stuff:
        if not os.path.exists('../Figures/' + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name']):
            os.makedirs('../Figures/' + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'])
        plt.savefig("../Figures/" + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'] + "/RIR.png")
    plt.show()

    plt.figure(figsize=(6.4*figure_size, 4.8*figure_size))
    for output in outputs:
        plt.plot(np.array(range(rir_start, rir_end))/Fs,
                 output['audio'][rir_start:rir_end],
                 label=output['label'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('RIR amplitude')
    plt.title('Early RIR')
    plt.legend()
    if save_stuff:
        plt.savefig("../Figures/" + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'] + "/Early RIR.png")
    plt.show()

    # exit()
    
 

    print('\tStarting EDCs')

    for output in outputs:
        try:
            output['EDC'] = EDC(output['audio'])
        except Exception as e:
            print('An error occurred during EDC.')
            print(e)

    print('\tComputed EDCs')

    EDC_durations = []
    for o in outputs:
        EDC_durations.append(len(o['EDC']))
    min_EDC_duration = min(min(EDC_durations), int(Fs * room_parameters['estimated T60'] / 4))


    plt.figure(figsize=(6.4 * figure_size, 4.8 * figure_size))
    for output in outputs:
        plt.plot(np.array(range(room_parameters['l.o.s. delay'], min_EDC_duration)) / Fs,
                 output['EDC'][room_parameters['l.o.s. delay']:min_EDC_duration],
                 label=output['label'])
    plt.title('log-time EDC difference, ' + room_parameters['subfolder name'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy difference (dB)')
    plt.xscale('log')
    plt.legend()
    if save_stuff:
        plt.savefig("../Figures/" + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'] + "/log-time EDC difference.png")
    plt.show()

    plt.figure(figsize=(6.4 * figure_size, 4.8 * figure_size))
    for output in outputs:
        plt.plot(np.array(range(room_parameters['l.o.s. delay'], min_EDC_duration)) / Fs,
                 output['EDC'][room_parameters['l.o.s. delay']:min_EDC_duration],
                 label=output['label'])
    plt.title('EDC, ' + room_parameters['subfolder name'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy difference (dB)')
    plt.legend()
    if save_stuff:
        plt.savefig("../Figures/" + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'] + "/EDC.png")
    plt.show()

    print('\tStarting NEDs')

    for output in outputs:
        try:
            output['short NED'] = ned.echoDensityProfile(output['audio'][:short_ned_end], short_ned_window)
            output['short NED'] = output['short NED'][:short_ned_end-2*short_ned_window]

            output['long NED'] = ned.echoDensityProfile(output['audio'][:long_ned_end], long_ned_window)
            output['long NED'] = output['long NED'][:long_ned_end-2*long_ned_window]
        except Exception as e:
            print('An error occurred during EDC.')
            print(e)
            output['short NED'] = None
            output['long NED'] = None

    print('\tComputed NEDs')

    plt.figure(figsize=(6.4*figure_size, 4.8*figure_size))
    for output in outputs:
        if output['short NED'] is not None:
            plt.plot(np.array(range(len(output['short NED'])))/Fs,
                     output['short NED'],
                     label=output['label'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Echo Density')
    plt.title('short NED')
    plt.legend()
    if save_stuff:
        plt.savefig("../Figures/" + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'] + "/short NED.png")
    plt.show()

    plt.figure(figsize=(6.4*figure_size, 4.8*figure_size))
    for output in outputs:
        if output['long NED'] is not None:
            plt.plot(np.array(range(len(output['long NED'])))/Fs,
                     output['long NED'],
                     label=output['label'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Echo Density')
    plt.title('long NED')
    plt.legend()
    if save_stuff:
        plt.savefig("../Figures/" + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'] + "/long NED.png")
    plt.show()

    # plt.figure(figsize=(6.4*figure_size, 4.8*figure_size))
    # for output in outputs:
    #     if 'nodes' in output.keys() and 'time' in output.keys():
    #         plt.scatter(output['nodes'],
    #                     output['time'],
    #                     label=output['label'])
    # plt.xlabel('Number of nodes')
    # plt.ylabel('Time (seconds)')
    # plt.title('Running time w.r.t. #nodes')
    # plt.legend()
    # if save_stuff:
    #     plt.savefig("../Figures/" + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'] + "/time to nodes.png")
    # plt.show()

    # plt.figure(figsize=(6.4*figure_size, 4.8*figure_size))
    # for output in outputs:
    #     if 'lines' in output.keys() and 'time' in output.keys():
    #         plt.scatter(output['lines'],
    #                     output['time'],
    #                     label=output['label'])
    # plt.xlabel('Number of lines')
    # plt.ylabel('Time (seconds)')
    # plt.title('Running time w.r.t. #lines')
    # plt.legend()
    # if save_stuff:
    #     plt.savefig("../Figures/" + folder_name + "/" + exp_name + "/" + room_parameters['subfolder name'] + "/time to lines.png")
    # plt.show()
    
