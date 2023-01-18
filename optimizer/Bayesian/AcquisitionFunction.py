import numpy as np
from abc import abstractmethod, ABC
from optimizer.Bayesian.GaussianModel import GaussianModel, BasicGaussianModel, PosteriorGaussianModel
from optimizer.Bayesian.Data import Data
import scipy.stats
from optimizer.Domain import Domain

class AcquisitionFunction(ABC):
    def setData(self, posteriorModel : GaussianModel, data : Data):
        self.posteriorModel = posteriorModel
        self.data = data

    def getPosteriorModel(self)->GaussianModel:
        return self.posteriorModel

    def getData(self)->Data:
        return self.data

    @abstractmethod
    def evaluate(self, x):
        pass 

class ExpectedImprovementAcquisition(AcquisitionFunction):
    def evaluate(self, x):
        weight = 0.01
        bestX, bestY = self.getData().getMinXY()
        stdev = np.sqrt(self.getPosteriorModel().covariance(x, x))
        if (stdev == 0):
            Z = 0
        else:
            Z = (bestY - self.getPosteriorModel().mean(x) - weight) / stdev

        improvement = stdev * (Z * scipy.stats.norm(0, 1).cdf(Z) + scipy.stats.norm(0, 1).pdf(Z))
        return improvement