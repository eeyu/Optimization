import numpy as np
from abc import abstractmethod, ABC
from optimizer.Bayesian.GaussianModel import GaussianModel, BasicGaussianModel, PosteriorGaussianModel
from optimizer.Bayesian.Data import Data
import scipy.stats
from optimizer.Domain import Domain
from optimizer.Bayesian.AcquisitionFunction import AcquisitionFunction
from visualizer import DebugMessage
from optimizer.Optimizer import Optimizer
from optimizer.CostEvaluator import CostEvaluator

class BayesianOptimizer(Optimizer):
    # Assuming exact measurements
    # for x in R^d space, domain is a list of d min/max
    # costFunction is f(x): R^d -> R
    def __init__(self, priorModel : GaussianModel, acquisitionFunction : AcquisitionFunction, costFunction : CostEvaluator, domain : Domain, initialSample):
        super().__init__(initialSample, costFunction)
        # initial assumptions
        self.priorModel = priorModel
        # Keeps track of data
        self.data = Data()
        self.posteriorModel = PosteriorGaussianModel(priorModel, self.data)

        # Set first sample
        initialCost = costFunction.getCost(initialSample)
        self.data.append(initialSample, initialCost)

        self.acquisitionFunction = acquisitionFunction
        acquisitionFunction.setData(self.posteriorModel, self.data)
        self.domain = domain
        self.costFunction = costFunction

        self.acquititionEvaluationAtStep = 0
        self.chosenLocationAtStep = 0

        self.iterations = 0

    def takeStepAndGetValueAndCost(self) -> tuple[np.ndarray, float]:
        self.takeStep()
        bestX, bestY = self.data.getMinXY()
        
        # self.debugMessage.appendMessage("EvaluatorMessage", bestDebugMessage)

        return bestX, bestY

    def takeStep(self):
        newX = self.selectPositionToEvaluate()
        newY = self.costFunction.getCost(newX)
        self.data.append(newX, newY)
        self.iterations += 1 

    def selectPositionToEvaluate(self):
        bestPosition = self.optimizeAcquisition(self.acquisitionFunction, self.domain, self.posteriorModel, self.data)
        return bestPosition

    # Brute force grid evaluation
    def optimizeAcquisition(self, acquisitionFunction : AcquisitionFunction, domain : Domain, posteriorModel : GaussianModel, data : Data):
        # Evaluate acquisition function over
        meshList, functionList = domain.createFunctionEvaluationOnList(acquisitionFunction.evaluate)
        bestIndex = np.argmax(functionList)
        bestPosition = meshList[bestIndex]
        self.debugMessage.appendMessage("bestPosition", bestPosition)
        self.acquititionEvaluationAtStep = functionList
        self.chosenLocationAtStep = (bestPosition, functionList[bestIndex])
        # self.debugMessage.appendMessage("Mesh", meshList)
        # self.debugMessage.appendMessage("AcquisitionFunction", functionList)
        return bestPosition

    def getPosterior(self):
        return self.posteriorModel


