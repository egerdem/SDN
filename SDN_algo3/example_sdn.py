#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:33:43 2021

@author: od0014
"""

import Geometry as geom
import Signal as sig
import Source as src
import Microphone as mic
import Simulation as sim
import numpy as np
from scipy.signal import butter
from scipy.io import savemat
import matplotlib.pyplot as plt
import timeit
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5

# create the input signal
Fs = 44100
# length = (int) (1 * Fs)
length = (int) (0.25 * Fs)
data = np.zeros(length, dtype = float)
data[0] = 1.0
signal = sig.Signal(Fs, data)


# make the room and choose its shape
room = geom.Room()
room.shape = geom.Cuboid(9,7,4)

#set wall filters and attenuation

# loop over 6 walls
nWalls = room.shape.nWalls

room.wallAttenuation = [0.89 for i in range(nWalls)]

# REMOVING W FILTERS TO COMPARE THE REF SDN, Ege
#butterworth lowpass filter at the end of walls
# filtOrder = 4
filtOrder = 1
# [b,a] = butter(filtOrder, 15000.0/(Fs/2), btype = 'low')

b = np.array([1.0, 0])
a = np.array([1.0, 0])

#couldn't find an easier way to make a 2D array in python
room.wallFilters = [[geom.WallFilter(filtOrder, b, a)
                     for j in range(nWalls-1)] for i in range(nWalls)]
# print(room.wallFilters[0])

# choose position of source 
srcPosition = geom.Point(4.5, 3.5, 2)
# create source at specified position, with input signal
source = src.Source(srcPosition, signal)


#choose position of mic
micPosition = geom.Point(2,2,1.5)
# create microphone at specified position
microphone = mic.Microphone(micPosition)


# size of each audio buffer, typically power of 2
frameSize = 8


"""
Run SDN for 1 test case
"""
nSamples = 5000
simulate = sim.Simulation(room, source, microphone, frameSize, nSamples)
output = simulate.run()
# plot the output
plt.figure()
plt.plot(output)
plt.show()
   
"""
"""
# Run SDN for many test cases and plot computation time
"""    
# how many samples of output do we want
nSamples = np.arange(100,length,5000)
nTime = np.zeros(len(nSamples), dtype = float)

# time the simulation
for i in range(len(nSamples)):
    # create the simulation
    simulate = sim.Simulation(room, source, microphone, frameSize, nSamples[i])
  
    # start the timer
    start = timeit.default_timer()
    # run the simulation
    output = simulate.run()
    #stop the timer
    stop = timeit.default_timer()
    #count interval
    nTime[i] = stop - start

    print('Time taken to simulate ', nSamples[i],' samples is '
          , np.round(nTime[i],3), ' seconds.')

seconds = np.linspace(0, 0.25, len(output))

output = simulate.run()
#plot the output
plt.figure()
plt.plot(seconds[:len(output)//2], output[:len(output)//2])
plt.show()

#plot and save resultant times
plt.plot(nSamples/Fs * 1000, nTime)
plt.xlabel('Length of RIR in ms')
plt.ylabel('Time taken for computation in s')
plt.grid()
# plt.savefig('../python-runtime.png')
plt.show()

#save data to compare with Matlab results
# mdic = {"output": output, "label" : "test_sdn"}
# savemat("../Data/sdn_test_python_buffer=" +str(frameSize) + ".mat", mdic)


"""