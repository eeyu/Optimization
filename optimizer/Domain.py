import numpy as np
from array import *
from abc import abstractmethod, ABC

class Domain(ABC):
    # min: dx1 min bounds
    # max: dx1 max bounds
    # resolution: resolution in each dim
    def __init__(self, min, max, resolution=10):
        # We construct the mesh representing the bounds
        dim = len(min)
        grid = np.linspace(min, max, resolution) # create intervals on each axis
        # TODO This step is pretty fucked up...fix it
        mesh=np.array(np.meshgrid(*grid)) # turn axis intervals into a mesh
        numPoints = int(mesh.size / dim)
        list = np.reshape(mesh.T, (numPoints, dim), 'F') # turn mesh into a list of numbers

        self.mesh = mesh
        self.list = list
        self.numPoints = numPoints
        self.dim = dim
        
    def getDomainList(self):
        return self.list

    # function: R^d -> R
    def createFunctionEvaluationsOnMesh(self, function, numpy=False):
        print("Domain: implement mesh generation in child class")
        pass

    # function: R^d -> R
    def createFunctionEvaluationOnList(self, functionToEval, numpy=False):
        if numpy:
            functionList = functionToEval(self.list)
        else:
            functionList = np.zeros((self.numPoints, self.dim))
            for i in range(self.numPoints):
                functionList[i] = functionToEval(self.list[i])

        return self.list, functionList

class Domain2D(Domain):
    def createFunctionEvaluationsOnMesh(self, function, numpy=False):
        pass

class Domain1D(Domain):
    def createFunctionEvaluationsOnMesh(self, function, numpy=False):
        return self.createFunctionEvaluationOnList(function, numpy)
