import numpy as np

class Data:
    def __init__(self):
        self.x = np.array([[]]) # this is nxd for domain R^d
        self.y = np.array([[]]) # this is nx1
    
    def append(self, x, y):
        if self.x.size == 0:
            self.x = np.array([x])
            self.y = np.array([y])
        else:
            self.x = np.append(self.x, [x], axis=0)
            self.y = np.append(self.y, [y], axis=0)

    def getMinXY(self):
        if len(self.y) > 0:
            bestIndex = np.argmin(self.y)
            bestY = self.y[bestIndex]
            bestX = self.x[bestIndex]
            return bestX, bestY
        return 0,0

    def getNumData(self):
        return len(self.x)