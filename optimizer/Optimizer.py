# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:11:22 2021

@author: Evan Yu
"""
import numpy as np;
from abc import ABC, abstractmethod
from dataclasses import dataclass
import optimizer.CostEvaluator as CostEvaluator
from visualizer.DebugMessage import DebugMessage



class Optimizer(ABC):
    def __init__(self, initialValue, costEvaluator : CostEvaluator):
        self.value = initialValue
        self.costEvaluator = costEvaluator

        self.stepCount = 0
        self.valueHistory = [initialValue]
        self.costHistory = np.array([costEvaluator.getCost(initialValue)])
        
        self.numFeatures = initialValue.size
        
        self.maxSteps = 1
        self.printEveryNSteps = 20
        self.convergenceThreshold = 0.0

        self.endNow = False
        
        self.debugMessage = DebugMessage()
    
    @abstractmethod
    def takeStepAndGetValueAndCost(self) -> tuple[np.array, float]:
        pass
    
    def step(self):
        self.debugMessage = DebugMessage()
        self.debugMessage.appendMessage("step", self.stepCount+1)
        
        self.costEvaluator.setOptimizerIteration(self.stepCount)
        value, cost = self.takeStepAndGetValueAndCost()
        self.value = value
        
        self.debugMessage.appendMessage("cost", cost)
        self.debugMessage.appendMessage("value", value)
        self.valueHistory.append(self.value)
        self.costHistory = np.append(self.costHistory, cost)
        self.stepCount += 1
        
        
    def getCurrentStateAndCost(self):
        return self.valueHistory[-1], self.costHistory[-1]
    
    def getFullHistory(self):
        return (self.valueHistory, self.costHistory)
        
    def hasReachedMinimum(self, convergenceThreshold):
        if len(self.costHistory) < 2:
            return False
        currentCost = self.costHistory[-1]
        lastCost = self.costHistory[-2]
        return abs(lastCost - currentCost) < convergenceThreshold
    
    def setOptimizationEndConditions(self, optimizationEndConditions):
        self.maxSteps = optimizationEndConditions.maxSteps
        self.convergenceThreshold = optimizationEndConditions.convergenceThreshold
        
    def hasReachedEndCondition(self):
        return (((self.maxSteps > 0) and (self.stepCount >= self.maxSteps)) or
                (self.hasReachedMinimum(self.convergenceThreshold)) or
                (self.endNow))
    
    def endEarly(self):
        self.endNow = True
    
    def optimizeUntilEndCondition(self, optimizationEndConditions):
        self.setOptimizationEndConditions(optimizationEndConditions)
        self.stepCount = 0
        while (not self.hasReachedEndCondition()):

            self.step()
            if (self.stepCount % self.printEveryNSteps == 0):
                self.printDebug()
    
    def setupOptimizer(self, optimizationEndConditions):
        self.setOptimizationEndConditions(optimizationEndConditions)
        self.stepCount = 0
    
    def optimizeNStepsOrUntilEndCondition(self, n):
        for i in range(n):
            if not self.hasReachedEndCondition():
                self.step()
                if (self.stepCount % self.printEveryNSteps == 0):
                    self.printDebug()

    def printDebug(self):
        print(self.debugMessage)
                
    def bindEndEarly(self, endEarly):
        self.endEarly = endEarly

@dataclass
class OptimizationEndConditions:
    maxSteps : int
    convergenceThreshold : float
    

    
    
    
