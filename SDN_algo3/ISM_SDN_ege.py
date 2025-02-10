import pyroomacoustics as pra
import Geometry as geom
import Signal as sig
import Source as src
import Microphone as mic
import Simulation as sim
import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import rrdecay as rr
import rt60
import pyroomacoustics.experimental as pra_exp

# create the input signal
Fs = 44100
t = 1
length = (int)(t * Fs)
# length = (int) (0.25 * Fs)
data = np.zeros(length, dtype=float)
data[0] = 1.0
signal = sig.Signal(Fs, data)

# size of each audio buffer, typically power of 2
frameSize = 8

# make the room and choose its shape
room = geom.Room()
# room.shape = geom.Cuboid(9,7,4)

room_parameters = {'width': 9, 'depth': 7, 'height': 4,
                   'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   'absorption': 0.2,
                   'air': {'humidity': 50,
                           'temperature': 20,
                           'pressure': 100},
                   'duration': 2.0}

# room_parameters =  {     'width': 9, 'height': 7, 'depth': 4,
#                          'source x': 4.5, 'source y': 3.5, 'source z': 2,
#                          'mic x': 2, 'mic y': 2, 'mic z': 1.5,
#                          'absorption': 0.2,
#                          'duration': 2.0}


room_parameters['reflection'] = 1 * np.sqrt(1 - room_parameters['absorption'])
room_parameters['duration'] = int(room_parameters['duration'] * Fs)

srcpos = np.array([room_parameters['source x'], room_parameters['source y'], room_parameters['source z']])
micpos = np.array([room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z']])
room_parameters['l.o.s. delay'] = int(np.ceil(np.linalg.norm(srcpos - micpos) * Fs / 343))
rir_start = int(max(0, room_parameters['l.o.s. delay'] - 10))
rir_end = rir_start + int(Fs / 20)

# Newer method for room initiation
# make the room and choose its shape
room = geom.Room()
room.shape = geom.Cuboid(room_parameters['width'],
                         room_parameters['depth'],
                         room_parameters['height'])

nWalls = room.shape.nWalls

# set wall filters and attenuation -- loop over 6 walls
# yeniden :
# room.wallFilters = [None] * 6
room.wallAttenuation = [room_parameters['reflection']] * 6

# eskiden:
# room.wallAttenuation = [0.9 for i in range(nWalls)]

# eskiden
# butterworth lowpass filter at the end of walls
# filtOrder = 4
filtOrder = 1

# [b,a] = butter(filtOrder, 15000.0/(Fs/2), btype = 'low')
b = np.array([1.0, 0])
a = np.array([1.0, 0])

# couldn't find an easier way to make a 2D array in python
room.wallFilters = [[geom.WallFilter(filtOrder, b, a)
                     for j in range(nWalls - 1)] for i in range(nWalls)]

# choose position of source
srcPosition = geom.Point(4.5, 3.5, 2)
# create source at specified position, with input signal
source = src.Source(srcPosition, signal)

# choose position of mic
micPosition = geom.Point(2, 2, 1.5)
# create microphone at specified position
microphone = mic.Microphone(micPosition)

outputs = []

"""
Run SDN
"""
nSamples = length
simulate = sim.Simulation(room, source, microphone, frameSize, nSamples)
audio = simulate.run()
audio = audio / np.max(np.abs(audio))

# ege order test. abort
# Print path lengths by order
# print("\nSDN Path Lengths by Reflection Order:")
# for order in sorted(simulate.path_lengths_by_order.keys()):
#     paths = simulate.path_lengths_by_order[order]
#     print(f"\nOrder {order} reflections:")
#     print(f"Number of paths: {len(paths)}")
#     print("Path lengths (meters):")
#     for length, count in paths:
#         print(f"  Length: {length:.3f}m, Reflection count: {count}")

    # Calculate average path length for this order
    # avg_length = sum(p[0] for p in paths) / len(paths)
    # print(f"Average path length for order {order}: {avg_length:.3f}m")
    # if order == 5:
    #     break

outputs.append({})
outputs[-1]['audio'] = audio
outputs[-1]['path_lengths'] = simulate.path_lengths_by_order  # Store path lengths for later use
outputs[-1]['label'] = "SDN"

run_pra = True
pra_order = 100
figure_size = 1.5
trim_others = True

if run_pra:
    print('\tStarting pyroomacoustics')

    room_dim = np.array([room_parameters['width'],
                         room_parameters['depth'],
                         room_parameters['height']])

    source_loc = np.array([room_parameters['source x'],
                           room_parameters['source y'],
                           room_parameters['source z']])
    mic_loc = np.array([room_parameters['mic x'],
                        room_parameters['mic y'],
                        room_parameters['mic z']])

    if room_parameters.get('air') is not None:
        print("not none")
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

# Inputs
sdn_rir = outputs[0]['audio']  # SDN RIR
# pra_rir = outputs[1]['audio']  # ISM RIR

# Compute RT60
# print(f"ISM rt60 theory sabine: {pra_room.rt60_theory()}")
# pra_room.measure_rt60()
# print(f"ISM rt60 measured: {pra_room.measure_rt60()}")

print("sdn_rir_rt60", rt60.measure_rt60(sdn_rir, Fs))
# print("pra_rir_rt60", rt60.measure_rt60(pra_rir, Fs))

# Ensure same length for comparison
# min_length = min(len(sdn_rir), len(pra_rir))
# sdn_rir = sdn_rir[:min_length]
# pra_rir = pra_rir[:min_length]

rir_sdn_early = sdn_rir[rir_start:rir_end]
# rir_ism_early = pra_rir[rir_start:rir_end]

# error = rr.calculate_rir_error(sdn_rir, pra_rir, Fs)
# error_early = rr.calculate_rir_error(rir_sdn_early, rir_ism_early, Fs)

# print(f"Error between RIRs: {error}")
# print(f"Error between early RIRs: {error_early}")

# Compute squared magnitudes
sdn_energy = np.sum(np.square(sdn_rir))
# pra_energy = np.sum(np.square(pra_rir))

# Print energy comparison
print("SDN RIR Energy:", sdn_energy)
# print("ISM RIR Energy:", pra_energy)
# print("Energy Ratio (SDN/ISM):", sdn_energy / pra_energy)

up_to_ms = 50
up_to_samples = int(Fs * up_to_ms / 1000)

# for output in outputs:
#     output['EDC'] = rr.EDC(output['audio'])

# EDC_sdn = outputs[0]['EDC']
# EDC_pra = outputs[1]['EDC']

# first_nonzero_index = np.nonzero(sdn_rir)[0][0]

# EDC_sdn_early = EDC_sdn[first_nonzero_index: first_nonzero_index + up_to_samples]
# EDC_pra_early = EDC_pra[first_nonzero_index: first_nonzero_index + up_to_samples]

# EDCerror = rr.calculate_edc_error_db(EDC_sdn, EDC_pra, start_time=0.0, end_time=0.05, fs=Fs)
# EDCerror_early = rr.calculate_edc_error_db(EDC_sdn_early, EDC_pra_early, start_time=0.0, end_time=0.05, fs=Fs)

# print(f"Error between EDCs: {EDCerror}")
# print(f"Error between early EDCs: {EDCerror_early}")

""" sdfgsdfgsdfgsdfgsdfgsdfhsfgh """
# Plotting

# Normalize for fair comparison
# sdn_rir_normalized = sdn_rir / np.max(np.abs(sdn_rir))
# pra_rir_normalized = pra_rir / np.max(np.abs(pra_rir))

# Frequency domain analysis
# sdn_spectrum = np.abs(np.fft.rfft(sdn_rir_normalized))
# pra_spectrum = np.abs(np.fft.rfft(pra_rir_normalized))
# frequencies = np.fft.rfftfreq(min_length, d=1 / Fs)

plt.figure(figsize=(6.4 * figure_size, 4.8 * figure_size))
for output in outputs:
    plt.plot(np.array(range(len(output['audio']))) / Fs,
             output['audio'],
             label=output['label'], alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('RIR amplitude')
plt.title('RIR')
plt.legend()

plt.figure(figsize=(6.4 * figure_size, 4.8 * figure_size))
for output in outputs:
    plt.plot(np.array(range(rir_start, rir_end)) / Fs,
             output['audio'][rir_start:rir_end],
             label=output['label'])
plt.xlabel('Time (seconds)')
plt.ylabel('RIR amplitude')
plt.title('Early RIR')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))

# Time-domain comparison
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(min_length) / Fs, sdn_rir_normalized, label='SDN RIR', alpha=0.7)
# plt.plot(np.arange(min_length) / Fs, pra_rir_normalized, label='ISM RIR', alpha=0.7)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Time-Domain RIR Comparison')
# plt.legend()

# Frequency-domain comparison
# plt.subplot(2, 1, 2)
# plt.semilogx(frequencies[frequencies <= 300], 20 * np.log10(sdn_spectrum[frequencies <= 300] + 1e-10),
#              label='SDN Spectrum', alpha=0.7)
# plt.semilogx(frequencies[frequencies <= 300], 20 * np.log10(pra_spectrum[frequencies <= 300] + 1e-10),
#              label='ISM Spectrum', alpha=0.7)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.title('Frequency-Domain RIR Comparison')
# plt.legend()

# plt.tight_layout()
# plt.show()

# EDC_durations = []
# for o in outputs:
#     EDC_durations.append(len(o['EDC']))
#
# room_parameters['estimated T60']  = 1.5
# min_EDC_duration = min(min(EDC_durations), int(Fs * room_parameters['estimated T60'] / 4))
#
# plt.figure(figsize=(6.4 * figure_size, 4.8 * figure_size))
# for output in outputs:
#     plt.plot(np.array(range(room_parameters['l.o.s. delay'], min_EDC_duration)) / Fs,
#              output['EDC'][room_parameters['l.o.s. delay']:min_EDC_duration],
#              label=output['label'])
# plt.title('EDC')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Energy difference (dB)')
# plt.legend()
# plt.show()
