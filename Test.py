import numpy as np
from array import *

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

x = np.array([1])
list = np.array([[]])
print(list.size)
print(list)
list = np.append(list, [x], axis=0)
print(list)