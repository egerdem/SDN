#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:47:49 2021

@author: od0014
"""
import numpy as np

class Signal:
    
    def __init__(self, sampleRate, data):
        self.sampleRate = sampleRate;
        self.length = len(data)
        self.data = data;
        
    
    def getFrame(self, startPos, frameSize):
        
        if (startPos > self.length):
            return np.zeros(frameSize, dtype=float)
        elif (startPos + frameSize < self.length):
            return self.data[startPos : startPos + frameSize]
        else:
            return np.append(self.data[startPos:], np.zeros(startPos + frameSize - self.length, dtype = float))
