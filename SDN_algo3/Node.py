#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:50:01 2021

@author: od0014
"""
import Geometry as geom
import ScatteringMatrix as scat
import numpy as np

    
class Node:
    """
    Class that implements each SDN node or wall junction
        
    NOTE : private variable names in Python generally start with _
    """
    
    def __init__(self, *args):
        
        self.wallFilter = []
        self.wallAttenuation = 0.0
        self.position = geom.Point(0,0,0)
        self._propLinesIn = []
        self._propLinesOut = []
        self.microphone_position = None
        
        if len(args) > 0:
            self.scatter = scat.ScatteringMatrix(args[0])
        else:
            self.scatter = scat.ScatteringMatrix("isotropic")
        
    
   
    # add incoming propagation lines
    def addPropLineIn(self, propLine):
        self._propLinesIn.append(propLine)
        
    
    
    # add outgoing propagation lines
    def addPropLineOut(self, propLine):
        self._propLinesOut.append(propLine)
        
        
    def createScatteringMatrix(self):
        self.scatter.createScatteringMatrix(len(self._propLinesIn))
        
    
    
    # process and output all incoming buffers from delay lines
    def getFramesOut(self, sourceFrame):
        
        assert (len(self._propLinesIn) == len(self._propLinesOut))
        
        # should be 5 for cuboid room
        N = len(self._propLinesOut)     
        
        # incoming buffers from all propagation lines to node
        framesIn = []
        for i in range(N):
            framesIn.append(self._propLinesIn[i].getCurrentFrame())
        
        
        
        # outgoing buffers from node to all outgoing propagation lines
        framesOut = []
        pressureOut = np.zeros(len(framesIn[0]))
        
        # Modified path tracking
        path_infos = []  # List to store PathInfo objects
        
        # loop over outgoing propagation lines
        for i in range(N):
                        
            # length equal to length of each incoming frame
            frameOut = np.zeros(len(framesIn[0]))
            
            # Track paths for each propagation direction
            for j in range(N):
                frameIn = framesIn[j].copy()
                frameIn += 0.5 * sourceFrame
                
                # Get distances
                source_to_node = self._propLinesIn[j].getStartJunction().position.getDistance(
                    self._propLinesIn[j].getEndJunction().position)
                
                node_to_node = self._propLinesOut[i].getStartJunction().position.getDistance(
                    self._propLinesOut[i].getEndJunction().position)
                
                node_to_mic = self._propLinesOut[i].getEndJunction().position.getDistance(
                    self.microphone_position)
                
                # Count reflections - each node-to-node propagation adds a reflection
                num_reflections = 1  # Start with 1 for first reflection
                if hasattr(self._propLinesIn[j], 'reflection_count'):
                    num_reflections += self._propLinesIn[j].reflection_count
                
                total_path = source_to_node + node_to_node + node_to_mic
                path_infos.append(PathInfo(total_path, num_reflections))
                
                # Update reflection count for outgoing propagation
                self._propLinesOut[i].reflection_count = num_reflections
                
                # Original scattering code
                frameOut += self.scatter.S[i,j] * frameIn
                    
            
            filteredFrameOut = self.wallFilter[i].processFrame(frameOut) * self.wallAttenuation
            #Ege - lfilter'I kaldırmaya çalışıyorum
            # filteredFrameOut = frameOut * self.wallAttenuation
            # print("Filtered frame out:", self.wallFilter[i].processFrame(frameOut))
            
            framesOut.append(filteredFrameOut)
            
            #see figure 6 in AES paper
            pressureOut += (2.0/N) * filteredFrameOut

        # Log the path lengths after first order reflections
        # print("Path lengths after first order reflections:", path_lengths)

        return (framesOut, pressureOut, path_infos)
    
    
    
    def pushNextFrameInPropLines(self, framesOut):
         
         N = len(self._propLinesOut)
         assert (len(framesOut) == N)
         for i in range(N):   
             self._propLinesOut[i].setNextFrame(framesOut[i])
             
          

            
                
    
        
        
    

class PathInfo:
    def __init__(self, length, num_reflections):
        self.length = length  # Total path length
        self.num_reflections = num_reflections  # Number of reflections
    
