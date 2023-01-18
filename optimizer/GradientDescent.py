import numpy as np;
from optimizer.Optimizer import Optimizer
from dataclasses import dataclass

class GradientDescentOptimizer(Optimizer):
    def __init__(self, initialValue, costEvaluator, optimizationParameters):
        super().__init__(initialValue, costEvaluator)
        self.optimizationParameters = optimizationParameters
        self.optimizationStepSize = optimizationParameters.optimizationStepSize
        self.gradientStepFactor = optimizationParameters.gradientStepFactor
        self.optimizationStepSizeScaling = optimizationParameters.optimizationStepSizeScaling
        self.scaleEveryNSteps = optimizationParameters.scaleEveryNSteps
        
    def findValueGradient(self):
        numDim = self.value.size
        valueGradient = np.zeros(numDim)
        currentCost = self.costHistory[-1]
        #construct gradient by sampling in every direction
        for i in range(numDim):
            valueTemp = np.copy(self.value)
            valueTemp[i] += self.gradientStepFactor * self.optimizationStepSize * self.optimizationParameters.weightedScaling[i]
            np.putmask(valueTemp, valueTemp<0, 0)
            costDim = self.costEvaluator.getCost(valueTemp)
            valueGradient[i] = currentCost - costDim
        gradientNorm = np.linalg.norm(valueGradient)
        valueGradient /= gradientNorm
        return valueGradient * self.optimizationParameters.weightedScaling
    
    def takeStepAndGetValueAndCost(self):
        if ((self.stepCount + 1) % self.scaleEveryNSteps == 0):
            self.optimizationStepSize *= self.optimizationStepSizeScaling
            
        valueGradientDirection = self.findValueGradient()
        valueStepVector = self.optimizationStepSize * valueGradientDirection
        value = self.value + valueStepVector
    #hack
        np.putmask(value, value<0, 0)
        cost = self.costEvaluator.getCost(value)
        return value, cost

       
    
    
@dataclass
class GDOptimizationParameters:
    optimizationStepSize : float
    gradientStepFactor : float
    optimizationStepSizeScaling : float
    scaleEveryNSteps : int
    weightedScaling : np.ndarray