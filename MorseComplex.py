import numpy as np
from UnionFind import *

"""
Coordinates system:
up: [0,-1]
down: [0,1]
left: [-1,0]
right:[1,0]
"""

class MorseComplex2D(object):
    def __init__(self, xComponentMatrix, yComponentMatrix, mandatoryMaxima):
        self.X = xComponentMatrix
        self.Y = yComponentMatrix
        self.mandatoryMaxima = mandatoryMaxima

    def getMandatoryFlowField(self):
        pass

    def getFlowField(self):
        pass

    def getMandatoryFlow(self, i, j):
        pass

    def getFlow(self, i, j):
        pass