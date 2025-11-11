#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:12:36 2021

@author: od0014
"""



class Reflection:
    
    """
    Class that determines wall node positions by calculating 1st order reflections
    """
    
    def __init__(self, sourcePos, micPos):
        
        self.sourcePos = sourcePos
        self.micPos = micPos
        
        
    def getReflectionAlongPlane(self, objPos, wall):
        
        return wall.plane.getPointReflection(objPos)
        
       
    def getNodePosition(self, wall):
        
            
        #find reflection position of source along wall
        sourceRefPos = self.getReflectionAlongPlane(self.sourcePos, wall)
        #node is at intersection of line connecting mic to source reflection and wall
        nodePos = wall.plane.findLineIntersection(sourceRefPos, self.micPos)
        
        # roundto a reasonable number of decimal places
        nodePos.x = round(nodePos.x, 5)
        nodePos.y = round(nodePos.y, 5)
        nodePos.z = round(nodePos.z, 5)
        
        return nodePos
            
            
            


    
   
    
    


        
            
        