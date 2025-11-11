#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:56:03 2021

@author: od0014
"""
import numpy as np
from scipy.signal import lfilter

class DelayLine:
    
    """
    Class implementing a delay line of 'length' samples. Pushes and pops a buffer
    at a time.
    
    """
    def __init__(self, length, frameSize):
        
        self.interpType = None
        
        # fractional delay line length
        # we want to evaluate y(n-frac)
        
        self.frac = (int(length) + 1) - length
        # integer delay line length
        self.length = int(length) + frameSize
        
        self.data = np.zeros(self.length, dtype = float)
        self.frameSize = frameSize
        
        
        if (self.length < self.frameSize):
            print("Length of delay line is ", self.length)
            raise Exception("Try smaller buffer size")
        
        
    def readBuffer(self):
        
        #interpolate for fractional delay lengths
        if (self.interpType == 'linear'):
            filt = self.linearInterpolation()
            outputData = filt
         
        elif (self.interpType == 'allpass'):
            filt = self.allpassInterpolation()
            outputData = filt
            
        else:    
            # extract output buffer
            outputData = self.data[-self.frameSize:]
        
        # return output buffer
        # the flip is needed so that oldest samples go out first
        # this is basically a FIFO queue
        return np.flip(outputData)
    
    
    def writeBuffer(self, inputData):
        
        if (inputData.size != self.frameSize):
            print(" Input data size is ", inputData.size, ' but frame size is ', self.frameSize)
            raise Exception("Buffer Sizes do not match")
            
        #shift data to the right
        self.data = np.roll(self.data, self.frameSize)
        
        # add buffer to start of delayLine
        # the flip is needed so that latest samples go in first
        # this is basically a FIFO queue
        self.data[:self.frameSize] = np.flip(inputData)
        
    
    # useful if delay line length is changed, or not an exact integer, 
    def linearInterpolation(self): 
        
        eta = self.frac
        y = lfilter(np.array([1-eta, eta]), np.array([1.0, 0.0]),
                                              self.data[:-self.frameSize])
        return y[:self.frameSize]
        
        
    def allpassInterpolation(self):
        
        eta = (1 - self.frac)/(1 + self.frac)
        y = lfilter(np.array([eta, 1.0]),np.array([1.0, eta]), 
                                                       self.data[:-self.frameSize])
        return y[:self.frameSize]
    


############################################################

class PropLine:
    
    """
    Class that implements propagation between two nodes

    """
    def __init__(self, nodeA, nodeB, frameSize, sampleRate, offset):
        
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.Fs = sampleRate
        self.c = 343    #speed of sound in air
        
        
        distance = self.nodeA.position.getDistance(self.nodeB.position)
        
        self.attenuation = (self.c/self.Fs) / distance  #1/r attenuation
        
        # setup the associated delay line with appropriate length
        
        # get rid of round while doing interpolation
        delayLength = round(self.Fs * (distance/self.c))
        # ask Enzo why this offset needs to be added
        if offset is not None:
            delayLength += offset
    

        self.delayLine = DelayLine(delayLength, frameSize)
        # interpolate delay line to have exact distance
        # self.delayLine.interpType = 'linear'
      
        self.reflection_count = 0  # Initialize reflection counter
        
    
    def getStartJunction(self):
        return self.nodeA
    
    
    def getEndJunction(self):
        return self.nodeB
    
    
    def setNextFrame (self, inputData):
        self.delayLine.writeBuffer(inputData)
        
    
    def getCurrentFrame(self):
        return self.delayLine.readBuffer() * self.attenuation
        
    
   
        