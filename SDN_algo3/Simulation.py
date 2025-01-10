#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:15:02 2021

@author: od0014
"""
import Node as node
import Reflection as reflect
import Propagate as prop
import numpy as np

class Simulation:
    
    def __init__(self, room, source, mic, frameSize, nSamples):
        self.room = room
        self.source = source
        self.microphone = mic
        self.frameSize = frameSize
        self.nSamples = nSamples
        self.verbose = False
        # Add dictionary to store path lengths by order
        self.path_lengths_by_order = {}  # {order: [(length, count)]}
        self.current_order = 0  # Track current reflection order
        
    
    def run(self):
        
        """ Setup all the nodes and propagation lines"""
        
        nWalls = self.room.shape.nWalls
        
        #create walls in the room, this returns a dict of walls
        
        walls = self.room.shape.setWallPosition()
            
        Fs = self.source.signal.sampleRate
        
        # create wall nodes
        
        nodes = []
        ref = reflect.Reflection(self.source.position, self.microphone.position)
        
        for wall in walls:
            # print(wall)
            wallNode = node.Node()
            wallNode.position = ref.getNodePosition(walls[wall])
            wallNode.microphone_position = self.microphone.position
            nodes.append(wallNode)
            # print(nodes)
            # for debugging
            # print ("Node position at " + wall + " = ", wallNode.position.x, 
            #         wallNode.position.y, wallNode.position.z)
            
            
            
        #add atteunation and wall filters
        
        for i in range(nWalls):
            nodes[i].wallAttenuation = self.room.wallAttenuation[i]
            nodes[i].wallFilter = self.room.wallFilters[i]

        # create all the propagation lines between wall nodes
        # print("nodes",nodes)
        for i in range(nWalls):
            for j in range(nWalls):
                
                if (i == j):
                    continue
                
                # The offset -frameSize means that we want a propagation line
                # with delaysample - frameSize. This is due to the way the
                # updating at the junction is made, which intrinsically
                # delays the output by frameSize samples.
                propLine = prop.PropLine(nodes[i], nodes[j], self.frameSize, Fs, -self.frameSize)
                           
                # initialize with zeros
                propLine.setNextFrame(np.zeros(self.frameSize, dtype = float))
                
                # line outgoing from node i is the same as
                nodes[i].addPropLineOut(propLine)
                # line incoming to line j
                nodes[j].addPropLineIn(propLine)
                
                propLine.attenuation = 1.0
                
            
        # create scattering matrix
        
        for i in range(nWalls):
            nodes[i].createScatteringMatrix()
        
                
        
        # create source propagation line
        
        sPropLines = []
        sDummyNode = node.Node()
        sDummyNode.position = self.source.position
        firstSourceFrame = self.source.signal.getFrame(0, self.frameSize)
        
        for i in range(nWalls):
            sPropLine = prop.PropLine(sDummyNode, nodes[i], self.frameSize, Fs, 0)
            sPropLine.setNextFrame(firstSourceFrame)
            
            sPropLines.append(sPropLine)
          
            
            
        # create microphone propagation line
        
        mPropLines = []
        mDummyNode = node.Node()
        mDummyNode.position = self.microphone.position
        
        for i in range(nWalls):
            
            mPropLine = prop.PropLine(nodes[i], mDummyNode, self.frameSize, Fs, 0)
            
            distSourceNode = nodes[i].position.getDistance(sDummyNode.position)
            distNodeMic = nodes[i].position.getDistance(mDummyNode.position)
            
            # set node to microphone attenuation, see paper for why this is chosen
            # We don't have (mPropLine.c ./ FS) at the numerator, because it is
            # already at the numerator of the attenuation of the
            # propLine between source and junction.
            mPropLine.attenuation = 1.0/(1.0 + distNodeMic/distSourceNode)
            mPropLines.append(mPropLine)
            
        
        
        # create direct path propagation line between source and mic
        
        smPropLine = prop.PropLine(sDummyNode, mDummyNode, self.frameSize, Fs, 0)
        smPropLine.setNextFrame(firstSourceFrame)       
        
        
        
        
        """ Run the simuation """
        
        nIter = int(np.ceil(self.nSamples/self.frameSize))
        output = np.zeros(self.frameSize*nIter, dtype = float)
        sampleNumber = 0
        count = 1;
  
        
        while (count <= nIter):
            
          
            if self.verbose:
                #print progress
                print("Running frame : ", count, " out of ", nIter)
            
            # initialize one buffer worth of output
            outputFrame = np.zeros(self.frameSize, dtype = float)
            
            framesOut = []
            current_paths = []  # Collect paths for current iteration
            
            for i in range(nWalls):
                
                # get incoming pressure from source
                sourceFrame = sPropLines[i].getCurrentFrame()
                
                # process scattering at wall node
                [frameOut, pressureFrame, path_infos] = nodes[i].getFramesOut(sourceFrame)
                framesOut.append(frameOut)
                current_paths.extend(path_infos)
                
                # push buffer containing pressure waves into propagation line connecting
                # wall node and microphone
                mPropLines[i].setNextFrame(pressureFrame)
                
                # get current output frame from microphone 
                # again, be careful about changing mPropLines' current frame
                outputFrame += mPropLines[i].getCurrentFrame().copy()
                
            # add direct path
            outputFrame +=  smPropLine.getCurrentFrame().copy()
            

            output[sampleNumber: sampleNumber + self.frameSize] = outputFrame 

            
            # prepare next buffer for processing
            sampleNumber += self.frameSize
            count += 1
            
            nextInputFrame = self.source.signal.getFrame(sampleNumber, self.frameSize)
            
            
            for i in range(nWalls):
                
                sPropLines[i].setNextFrame(nextInputFrame)
                nodes[i].pushNextFrameInPropLines(framesOut[i])
            
            
            smPropLine.setNextFrame(nextInputFrame)
            
            # Store paths for current order
            if current_paths:
                for path_info in current_paths:
                    order = path_info.num_reflections
                    if order not in self.path_lengths_by_order:
                        self.path_lengths_by_order[order] = []
                    self.path_lengths_by_order[order].append(
                        (path_info.length, path_info.num_reflections)
                    )
            
            # Increment order counter
            self.current_order += 1
            
            # Enzo clarified this, the outgoing propagation lines of one node are the 
            # incoming propagation lines of another. So just updating the outgoing propagation 
            # lines automatically updates the incoming ones. Python passes by reference, so it is okay.
            
        
        # we are finally done 
        return output[:self.nSamples]
        
        
        
        
        
        
            
            
            
            
            
            
            
                
                
                
        
        
        
        