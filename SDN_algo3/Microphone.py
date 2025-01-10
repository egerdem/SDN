#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:51:34 2021

@author: od0014
"""

class Microphone:
    
    
    def __init__(self, position):
        self.directivity = 1.0
        # heading is the angle formed between the x-axis and the source's 
        # axis. Angles are anticlockwise.
        self.heading = 0.0  
        self.position = position
        
    