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
        self.nRows, self.nCols = self.X.shape

        self.maxFlow = self._computeMandatoryFlowField(mandatoryMaxima)

    def _computeMandatoryFlowField(self, mandatoryMaxima):
        maxIndex = self.nRows*self.nCols

        # Offset the maxFlow indices so as not to collide with any
        # existing ids, we'll remove this offset at the end
        self.maxFlow = mandatoryMaxima
        self.maxFlow += maxIndex

        # Initialize the union-find data structure so we can
        # short-circuit the algorithm
        self.uf = UnionFind()
        for i in np.unique(self.maxFlow):
            self.uf.MakeSet(i)

        # Initialize the union-find with items where we know the
        # solution
        for row in range(self.nRows):
            for col in range(self.nCols):
                idx = row*self.nCols+col
                self.uf.MakeSet(idx)
                if self.maxFlow[row, col] >= maxIndex:
                    self.uf.Union(self.maxFlow[row, col], idx)

        # Now compute the rest
        for row in range(self.nRows):
            for col in range(self.nCols):
                idx = row*self.nCols+col
                touched = set()
                destination = self.trace(row, col, touched)
                self.uf.Union(idx, destination)
                self.maxFlow[row, col] = destination

        self.maxFlow -= maxIndex
        return self.maxFlow

    def trace(self, row, col, touched):
        idx = row*self.nCols+col
        if idx in touched:
            return idx
        touched.add(idx)
        flow = self.uf.Find(idx)
        if flow >= self.nRows*self.nCols:
            return flow
        else:
            moveX = self.X[row, col]
            moveY = self.Y[row, col]
            newRow = row + moveY
            newCol = col + moveX
            if newRow == row and newCol == col:
                return idx
            return self.trace(newRow, newCol, touched)

    def _computeFlowField(self):
        pass