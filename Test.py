import numpy as np
from array import *
from optimizer.Domain import Domain

x = np.array([1])
y = np.array([3])

# griddy = np.linspace(x, y, 40).T

# print(*griddy)

# dim = len(x)
# meshTemp = np.meshgrid(*griddy)
# # print(meshTemp)
# mesh=np.array(meshTemp)
# # print(mesh)
# # print(mesh.T.shape)
# # print(mesh)
# numPoints = int(mesh.size / dim)
# list = np.reshape(mesh.T, (numPoints, dim), 'F')
resolution = 40
min = x
max = y
dim = len(min)
grid = np.linspace(min, max, resolution).T # create intervals on each axis
# TODO This step is pretty fucked up...fix it
mesh=np.array(np.meshgrid(*grid)) # turn axis intervals into a mesh
numPoints = int(mesh.size / dim)
list = np.reshape(mesh.T, (numPoints, dim), 'F') # turn mesh into a list of numbers

# self.mesh = mesh
# self.list = list
# self.numPoints = numPoints
# self.dim = dim

# domain = Domain(x, y, 40)
# list = domain.getDomainList()
print(list.shape)

# x = np.array([1,3])
# y = np.array([3,6])
# z = range(4)
# grid = [x, y, z]

# griddy = np.linspace(x, y, 5).T

# # print(*griddy)

# dim = len(x)
# mesh=np.array(np.meshgrid(*griddy))
# # print(mesh.T.shape)
# # print(mesh.T)
# numPoints = int(mesh.size / dim)
# list = np.reshape(mesh.T, (numPoints, dim), 'F')
# # print(list)


# class Foo:
#     def printIt(self, x):
#         print(x)

# def callFunction(function):
#     function("hello")

# foo = Foo()
# callFunction(foo.printIt)

# x = np.array([1])
# list = np.array([[]])
# print(list.size)
# print(list)
# list = np.append(list, [x], axis=0)
# print(list)