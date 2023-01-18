from optimizer.Bayesian.BayesianOptimizer import BayesianOptimizer
from optimizer.Domain import Domain
from optimizer.Bayesian.GaussianModel import GaussianModel, BasicGaussianModel
import numpy as np
from optimizer.Bayesian.AcquisitionFunction import AcquisitionFunction, ExpectedImprovementAcquisition
from optimizer.CostEvaluator import CostEvaluator
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from visualizer.BayesianVisualizer1D import BayesianVisualizer1D

class ParaboloidCostEvaluator(CostEvaluator):
    def getCost(self, value):
        return value * value

priorModel = BasicGaussianModel(stdev=0.1, scale=0.5)
acquisitionFunction = ExpectedImprovementAcquisition()
costFunction = ParaboloidCostEvaluator()
domain = Domain(min=[-5], max=[5], resolution=20)
initialPoint = np.array([3])

optmizer = BayesianOptimizer(priorModel=priorModel, acquisitionFunction=acquisitionFunction, costFunction=costFunction, domain=domain, initialSample=initialPoint)
# for i in range(10):
#     optmizer.step()
#     optmizer.printDebug()
plotter = BayesianVisualizer1D(costFunction, domain, optmizer)

if __name__ == '__main__':
    import sys
    plotter.visualize()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    

