import numpy as np
from abc import abstractmethod, ABC
from optimizer.Bayesian.Data import Data

class VectorToolbox:
    # x represents either a list of values in R^d or a single value in R^d
    # The first is represented by a matrix and the second by a vector
    def isSingleton(x : np.ndarray):
        if len(x.shape) > 1:
            return False
        else:
            return True

    # assume x is some list of values, expressed in matrix form. the outer layer is the list.
    def getNumEntries(x : np.ndarray):
        if VectorToolbox.isSingleton(x):
            return 1
        else:
            return x.shape[0]

    # get dimension d of domain R^d
    def getDomainDimensions(x : np.ndarray):
        if VectorToolbox.isSingleton(x):
            return x.shape[0]
        else:
            return x.shape[1]

# assuming operation on some x in R^d.
# x may come in the form of a single value 1xd or a list of values nxd
# mean, covariance functions must operate on either a list or singleton 
class GaussianModel(ABC):
    # x can be a single object or a list
    def mean(self, x) -> np.ndarray:
        if VectorToolbox.isSingleton(x):
            return np.array([self.meanFunction(x)])
        else:
            numEntries = VectorToolbox.getNumEntries(x)
            meanReturn = np.zeros((numEntries, 1))
            for i in range(numEntries):
                meanReturn[i] = self.meanFunction(x[i])
            return meanReturn

    def covariance(self, x1, x2) -> np.ndarray:
        # make singleton
        
        if VectorToolbox.isSingleton(x1) and VectorToolbox.isSingleton(x2):
            return np.array([self.covarianceFunction(x1, x2)])
        # make vector n x 1
        elif (not VectorToolbox.isSingleton(x1)) and VectorToolbox.isSingleton(x2):
            numEntries = VectorToolbox.getNumEntries(x1)
            covarianceReturn = np.zeros((numEntries, 1))
            for i in range(numEntries):
                covarianceReturn[i] = self.covarianceFunction(x1[i], x2)
            return covarianceReturn
        # make vector 1 x n
        elif VectorToolbox.isSingleton(x1) and (not VectorToolbox.isSingleton(x2)):
            numEntries = VectorToolbox.getNumEntries(x2)
            covarianceReturn = np.zeros((1, numEntries))
            for i in range(numEntries):
                covarianceReturn[0][i] = self.covarianceFunction(x1, x2[i])
            return covarianceReturn
        # make matrix n x n
        else: # np.shape(x1) > 1 and np.shape(x2) > 1
            numEntries = VectorToolbox.getNumEntries(x1)
            covarianceReturn = np.zeros((numEntries, numEntries))
            for i in range(numEntries):
                for j in range(numEntries):
                    covarianceReturn[i][j] = self.covarianceFunction(x1[i], x2[j])
            return covarianceReturn

    # function acts on singleton
    @abstractmethod
    def covarianceFunction(self, x1, x2):
        pass

    # function acts on singleton
    @abstractmethod
    def meanFunction(self, x):
        pass

class BasicGaussianModel(GaussianModel):
    def __init__(self, stdev, scale):
        self.scale = scale
        self.stdev = stdev

    def covarianceFunction(self, x1, x2):
        cov = self.stdev*self.stdev * np.exp(-np.square(np.linalg.norm(x2 - x1)) / (2 * self.scale * self.scale))
        if cov < 0:
            return 0
        return cov

    def meanFunction(self, x):
        return np.zeros(VectorToolbox.getDomainDimensions(x))

class PosteriorGaussianModel(GaussianModel):
    def __init__(self, prior : GaussianModel, data : Data):
        self.prior = prior
        self.data = data

    def getData(self):
        return self.data

    def covarianceFunction(self, x1, x2):
        if self.data.getNumData() == 1:
            covInverse = np.array([np.reciprocal(self.prior.covariance(self.data.x, self.data.x))])
        else:
            covInverse = np.linalg.inv(self.prior.covariance(self.data.x, self.data.x))

        posteriorCovariance = (
            self.prior.covariance(x1, x2) - 
            self.prior.covariance(x1, self.data.x).dot(
            covInverse).dot(
            self.prior.covariance(self.data.x, x2))
            )
        if posteriorCovariance < 0:
            return 0
        return posteriorCovariance    

    def meanFunction(self, x):
        if self.data.getNumData() == 1:
            covInverse = np.array([np.reciprocal(self.prior.covariance(self.data.x, self.data.x))])
        else:
            covInverse = np.linalg.inv(self.prior.covariance(self.data.x, self.data.x))
        posteriorMean = (
            self.prior.mean(x) + 
            self.prior.covariance(x, self.data.x).dot( 
            covInverse).dot( 
            (self.data.y - self.prior.mean(self.data.x)))
            )
        return posteriorMean