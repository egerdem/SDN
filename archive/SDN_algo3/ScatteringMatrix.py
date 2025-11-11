#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:19:17 2021

@author: od0014

Class implementing scattering matrix
"""

import numpy as np

class ScatteringMatrix:
    
    def __init__(self, type):
       
        self.type = type 
        # size of matrix (square)
        self.N = 1
        # scattering matrix
        self.S = np.zeros((self.N,self.N))
        
        
    
    def createScatteringMatrix(self, N):
        
        self.N = N
        self.S = np.zeros((N,N), dtype = float)
        
        if self.type == "isotropic":
            self.createIsotropicScattering()
    
    
    def createIsotropicScattering(self):
        
        for i in range(self.N):
            for j in range(self.N):
                # diagonal of scattering matrix
                if i == j:
                    self.S[i,j] = 2.0/self.N - 1
                else:
                    self.S[i,j] = 2.0/self.N
                    
            
                        