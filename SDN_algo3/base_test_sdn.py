#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:38:27 2021

@author: od0014

Run SDN unit tests
"""
import math
from src import Geometry as geom 
from src import Reflection as ref
from src import Node as node
from src import Propagate as prop
from src import Source as src
from src import Microphone as mic
from src import Signal as sig
from src import Simulation as sim
import numpy as np

############################################
# test 1 to check geometry

room = geom.Room()
room.shape = geom.Cuboid(1,1,1)

walls = room.shape.setWallPosition()

srcPos = geom.Point(0.5,0.5,0.5)
micPos = geom.Point(0.5,0.5,0.5)
reflect = ref.Reflection(srcPos, micPos)

refPos = dict()
refPos['floor'] = np.array([0.5,0.0,0.5])   # FACE 1 in Enzo's code
refPos['ceiling'] = np.array([0.5,1,0.5])   # Face 3 in Enzo's code
refPos['left'] = np.array([0, 0.5, 0.5])    # Face 4 in Enzo's code
refPos['right'] = np.array([1, 0.5, 0.5])   # Face 2 in Enzo's code
refPos['front'] = np.array([0.5,0.5, 1])    # Face 5 in Enzo's code
refPos['back'] = np.array([0.5, 0.5, 0])    # Face 6 in Enzo's code


for wall in walls:
    nodePos = reflect.getNodePosition(walls[wall])
    # print(wall, nodePos.x, nodePos.y, nodePos.z)
    # print(np.array([nodePos.x, nodePos.y, nodePos.z]) == refPos[wall])
    assert(np.sum(np.array([nodePos.x, nodePos.y, nodePos.z]) == refPos[wall]) == 3);
    
print('Test 1 passed!')

###############################################
# test 2 to check geometry

room = geom.Room()
room.shape = geom.Cuboid(1,1,1)

walls = room.shape.setWallPosition()

srcPos = geom.Point(0.25,0.5,0.5)
micPos = geom.Point(0.75,0.5,0.5)
reflect = ref.Reflection(srcPos, micPos)

refPos = dict()
refPos['floor'] = np.array([0.5,0.0,0.5])   # FACE 1 in Enzo's code
refPos['ceiling'] = np.array([0.5,1,0.5])   # Face 3 in Enzo's code
refPos['left'] = np.array([0, 0.5, 0.5])    # Face 4 in Enzo's code
refPos['right'] = np.array([1, 0.5, 0.5])   # Face 2 in Enzo's code
refPos['front'] = np.array([0.5,0.5, 1])    # Face 5 in Enzo's code
refPos['back'] = np.array([0.5, 0.5, 0])    # Face 6 in Enzo's code


for wall in walls:
    nodePos = reflect.getNodePosition(walls[wall])
    assert(np.sum(np.array([nodePos.x, nodePos.y, nodePos.z]) == refPos[wall]) == 3);
    
print('Test 2 passed!')


##################################################

# test 3 to check geometry

room = geom.Room()
room.shape = geom.Cuboid(1,1,1)

walls = room.shape.setWallPosition()

srcPos = geom.Point(0.2,0.2,0.2)
micPos = geom.Point(0.8,0.8,0.8)
reflect = ref.Reflection(srcPos, micPos)

refPos = dict()
refPos['floor'] = np.array([0.32,0.0,0.32])       # FACE 1 in Enzo's code
refPos['ceiling'] = np.array([0.68,1,0.68])   # Face 3 in Enzo's code
refPos['left'] = np.array([0, 0.32, 0.32])        # Face 4 in Enzo's code
refPos['right'] = np.array([1, 0.68, 0.68])   # Face 2 in Enzo's code
refPos['front'] = np.array([0.68,0.68, 1])    # Face 5 in Enzo's code
refPos['back'] = np.array([0.32, 0.32, 0])        # Face 6 in Enzo's code


for wall in walls:
    nodePos = reflect.getNodePosition(walls[wall])
    # print(wall, nodePos.x, nodePos.y, nodePos.z)
    # print(np.array([nodePos.x, nodePos.y, nodePos.z]) == refPos[wall])
    assert(np.sum(np.array([nodePos.x, nodePos.y, nodePos.z]) == refPos[wall]) == 3);
    
print('Test 3 passed!')

#################################################

# SDN test 1

nodeA = node.Node()
nodeA.position = geom.Point(0,0,0)
nodeB = node.Node()
nodeB.position = geom.Point(0.02,0,0)

distance = nodeA.position.getDistance(nodeB.position)
assert(distance == 0.02)

c = 343
Fs = 44100
frameSize = 1
attenuation = (c/Fs)/distance
delay = distance/c
latency = round(delay*Fs)
assert(latency == 3)

propLine = prop.PropLine(nodeA, nodeB, frameSize, Fs, 0)
assert(propLine.getStartJunction() == nodeA)
assert(propLine.getEndJunction() == nodeB)
assert(propLine.delayLine.length == latency + frameSize)


propLine.setNextFrame(np.array([1.0]))
assert(propLine.getCurrentFrame() == 0)
propLine.setNextFrame(np.array([2.0]))
assert(propLine.getCurrentFrame() == 0)
propLine.setNextFrame(np.array([3.0]))
assert(propLine.getCurrentFrame() == 0)
propLine.setNextFrame(np.array([-1.0]))
assert(propLine.getCurrentFrame() == 1.0*attenuation)
propLine.setNextFrame(np.array([-1.0]))
assert(propLine.getCurrentFrame() == 2.0*attenuation)
propLine.setNextFrame(np.array([-1.0]))
assert(propLine.getCurrentFrame() == 3.0*attenuation)
propLine.setNextFrame(np.array([-1.0]));
assert(propLine.getCurrentFrame() == -1.0*attenuation)

print("Test 4 passed!")

##################################################

# try with different frame size
nodeA = node.Node()
nodeA.position = geom.Point(0,0,0)
nodeB = node.Node()
nodeB.position = geom.Point(0.02,0,0)
frameSize = 2

propLine = prop.PropLine(nodeA, nodeB, frameSize, Fs, 0)
assert(propLine.getStartJunction() == nodeA)
assert(propLine.getEndJunction() == nodeB)
assert(propLine.delayLine.length == latency + frameSize)


propLine.setNextFrame(np.array([1.0, 2.0]))
res = propLine.getCurrentFrame().copy()
assert(np.array_equal(res,np.array([0,0])))


propLine.setNextFrame(np.array([3.0, -1.0]))
res = propLine.getCurrentFrame().copy()
assert(np.array_equal(res, np.array([0, 1.0*attenuation])))


propLine.setNextFrame(np.array([-1 , -1]))
res = propLine.getCurrentFrame().copy()
assert(np.array_equal(res, np.array([2.0*attenuation, 3.0*attenuation])))

print("Test 5 passed!")

###########################################
# SDN test 2, computation by hand

Fs = 44100
c = 343

minDist = c/Fs
room = geom.Room()
room.shape = geom.Cuboid(8 * minDist, 8 * minDist, 1000)

nWalls = room.shape.nWalls

room.wallAttenuation = [1.0 for i in range(nWalls)]

filtOrder = 1
b = np.array([1.0,0])
a = np.array([1.0,0])
room.wallFilters = [[geom.WallFilter(filtOrder, b, a)  
                     for j in range(nWalls-1)] for i in range(nWalls)]


position = geom.Point(6 * minDist, 5 * minDist, 500)
signal = sig.Signal(Fs, np.r_[1, np.zeros(8820)])
source = src.Source(position, signal)


position = geom.Point(3 * minDist, 5 * minDist, 500)
microphone = mic.Microphone(position)


frameSize = 1
Nsamples = 18
simulate = sim.Simulation(room, source, microphone, frameSize, Nsamples)
output = simulate.run()


cmp = np.zeros(18);
cmp[3] = 1/3;
cmp[7] = 1/7;
cmp[9] = 1/9;
cmp[10] = 1/(2*math.sqrt(5**2+1.5**2));
cmp[6] = 1/(2*math.sqrt(3**2+1.5**2));
cmp[11] = 2/(15*math.sqrt(3**2+1.5**2));
cmp[10] = cmp[10]+1/20;
cmp[13] = 1/15;
cmp[15] = 2/(15*math.sqrt(5**2+1.5**2));
cmp[13] = cmp[13]+1/20;
cmp[16] = 2/(5*7*math.sqrt(1.5**2+5**2));
cmp[16] = cmp[16]+1/(10*math.sqrt(1.5**2+5**2));
cmp[16] = cmp[16]+1/(10*math.sqrt(1.5**2+3**2));
cmp[13] = cmp[13]+2/(5*7*math.sqrt(1.5**2+3**2));
cmp[14] = 1/60;
cmp[17] = cmp[17]+1/(5*7)*(-3/5);
cmp[16] = cmp[16]+1/(10*math.sqrt(1.5**2+3**2))*(-3/5);
cmp[16] = cmp[16]+1/(10*math.sqrt(1.5**2+3**2))*(-3/5);
cmp[15] = cmp[15]+2/75;


print("Output should be ", cmp)
print()
print("Output is ", output)

assert(np.abs(np.sum(output - cmp)) < 1e-3)

print("Test 6 passed!")