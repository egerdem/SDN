#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:31:41 2021

@author: od0014
"""
import math
import numpy as np
from scipy.signal import lfilter


class Point:
    """
    Class that defines a point in 3D cartesian coordinate system
    """
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def getDistance(self, p):
        
        """ 
        Returns the Euclidean distance between two 3D positions in 
        cartesian coordinates
        """
        dis = math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2 + (self.z - p.z) ** 2)
        return dis
    
    def subtract(self, p):
        
        return np.array([self.x - p.x, self.y - p.y, self.z - p.z], dtype = float)
    
    def equals(self,p):
        
        if ((self.x == p.x) and (self.y == p.y) and (self.z == p.z)):
            return True
        else:
            return False

    

class Plane:

    """
    Class and helper functions defining a 3D plane
    """

    def __init__(self):

        #plane represented by ax + by + cz + d = 0 and its normal vector

        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.normal = np.zeros(3, dtype = float)

    def setPlaneFromPoints(self, posA, posB, posC):
        #posA, posB and posC are 3 points on a plane

        #find vector normal to the plane
        arr1 = posB.subtract(posA)
        arr2 = posC.subtract(posA)
        self.normal = np.cross(arr1, arr2)

        assert np.dot(self.normal, arr1) == 0.0, "normal vector not right"

        #scalar component
        self.d = np.dot(-self.normal, [posA.x, posA.y, posA.z])
        self.a = self.normal[0]
        self.b = self.normal[1]
        self.c = self.normal[2]


    def getPointReflection(self, point):
        # get reflection of a point along the plane

        # equation of line from (x1,y1,z1) to where it intersects plane
        # (x - x1) / a = (y - y1) / b = (z - z1) / c = k
        # replace x = ak + x1 etc in ax + by + cz + d = 0 to find k
        k = -(self.a * point.x + self.b * point.y + self.c * point.z +
             self.d) / (self.a ** 2 + self.b ** 2 + self.c ** 2)

        # where line from point intersects plane, is the midpoint between (x1,y1,z1)
        # and its reflection
        refPos = Point(0.0, 0.0, 0.0)
        refPos.x = 2 * (self.a * k + point.x)  - point.x
        refPos.y = 2 * (self.b * k + point.y)  - point.y
        refPos.z = 2 * (self.c * k + point.z)  - point.z

        return refPos


    def findLineIntersection(self, posA, posB):

        #two points are enough to define a line
        #equation of a line is (x-x1)/l = (y-y1)/m = (z-z1)/n = k
        l = posB.x - posA.x
        m = posB.y - posA.y
        n = posB.z - posA.z

        # replace x with kl + x1 etc and plug into ax + by + cz + d = 0 to find k
        k = -(self.a * posA.x + self.b * posA.y + self.c * posA.z +
              self.d) / (self.a * l + self.b * m + self.c * n)

        # plug in value of k into x = kl+x1 etc to find point of intersection
        interPos = Point(0.0, 0.0, 0.0)
        interPos.x = k*l + posA.x
        interPos.y = k*m + posA.y
        interPos.z = k*n + posA.z

        return interPos
        
        
        
        
class Room:
    """
    Class defining a room with some propoerties that can be controlled
    """
    def __init__(self):
        self.shape = ''
        self.wallAttenuation = []   #this is a list
        self.wallFilters = dict()   #this is a dictionary
    
    
    
    
    
class Cuboid:
    """
    Class defining a cuboid room with dimensions and wall positions
    """
    
    def __init__(self, x, y, z):
        self.name ='cuboid'
        self.nWalls = 6
        self.walls = {}
        self.x = x
        self.y = y
        self.z = z
        
        
    def setWallPosition(self):
        
        #allocate 3 points specifying each wall in room - 3 points are enough to define a plane
        
        # self.walls['floor'] = Wall(Point(0,0,0), Point(self.x,0,0),Point(0,0,self.z))
        # self.walls['ceiling'] = Wall(Point(0,self.y,0), Point(0, self.y, self.z),
        #                          Point(self.x, self.y, 0))
        # self.walls['left'] = Wall(Point(0,0,0), Point(0, self.y, 0),
        #                          Point(0,0,self.z))
        # self.walls['right'] = Wall(Point(self.x, 0, 0), Point(self.x, self.y, 0),
        #                          Point(self.x, self.y, self.z))
        # self.walls['front'] = Wall(Point(0,0,self.z), Point(0, self.y, self.z),
        #                          Point(self.x, self.y ,self.z))
        # self.walls['back'] = Wall(Point(0,0,0), Point(0, self.y, 0),
        #                          Point(self.x, self.y, 0))

        self.walls['back'] = Wall(Point(0, 0, 0), Point(self.x, 0, 0), Point(0, 0, self.z))
        self.walls['front'] = Wall(Point(0, self.y, 0), Point(0, self.y, self.z), Point(self.x, self.y, 0))
        self.walls['left'] = Wall(Point(0, 0, 0), Point(0, self.y, 0), Point(0, 0, self.z))
        self.walls['right'] = Wall(Point(self.x, 0, 0), Point(self.x, self.y, 0), Point(self.x, self.y, self.z))
        self.walls['ceiling'] = Wall(Point(0, 0, self.z), Point(0, self.y, self.z), Point(self.x, self.y, self.z))
        self.walls['ground'] = Wall(Point(0, 0, 0), Point(0, self.y, 0), Point(self.x, self.y, 0))
        
        for key in self.walls:
            self.walls[key].setPlaneCoefficients()
            
        return self.walls
 
        
 

class Wall:
    
    """
    Class defining a wall in a room, which represents a plane in 3D 
    """
    
    def __init__(self, posA, posB, posC):
        self.posA = posA
        self.posB = posB
        self.posC = posC
        self.plane = Plane()
        
    
    def setPlaneCoefficients(self):
        self.plane.setPlaneFromPoints(self.posA, self.posB, self.posC)
        
        


class WallFilter:
    
    """
    Class that defines the wall filter coefficients
    """
    
    def __init__(self, order, b, a):
        self.order =  order
        self.b = b
        self.a = a
        
    def setCoefficients(self,b,a):
        self.b = b
        self.a = a
        
        
    def processFrame(self, inputBuffer):
        """

        Parameters
        ----------
        inputBuffer :numpy array of floats

        Returns
        -------
        outputBuffer :  filtered array of float.

        """
        
        return lfilter(self.b, self.a, inputBuffer)
        
    
    

        

            
        
    
    
        
        
        
    