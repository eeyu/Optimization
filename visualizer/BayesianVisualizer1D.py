# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:51:38 2021

@author: Evan Yu
"""
import time

from abc import ABC, abstractmethod
from cmath import sin
import math
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from PyQt5.QtWidgets import QSlider

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget
    
from visualizer.QSlider import Slider
from optimizer.Bayesian.BayesianOptimizer import BayesianOptimizer
from optimizer.CostEvaluator import CostEvaluator
from optimizer.Domain import Domain


class BayesianVisualizer1D():
    def __init__(self, costEvaluator : CostEvaluator, domain : Domain, optimizer : BayesianOptimizer):
        self.app = QtGui.QApplication([])

        self.setDefaultOptions()
        self.costEvaluator = costEvaluator       
        self.domain = domain
        self.optimizer = optimizer
        self.domainX = self.domain.getDomainList().flatten()

        self.sampleXList = []
        self.sampleYList = []

        optimizer.step()
       
       # self.createEverything()
       
    def setDefaultOptions(self):
        self.setPlotRange(xMax = 15, yMax = 15)
        self.setResolution(200)
       
    def setPlotRange(self, xMax, yMax):
        self.xMax = xMax
        self.yMax = yMax
        self.avgGridSize = (xMax + yMax) / 2.0
        
    def setResolution(self, resolution):
        self.resolution = resolution
        
    def visualize(self):
        self.wMain = pg.LayoutWidget()
        self.wMain.setWindowTitle('2d cost function')
        
        # cost
        self.costPlot = self.createCostPlot()
        # acquisition
        self.acquisitionPlot = self.createAcquisitionPlot()
        # next button
        self.createButtons()

        self.acqLabel = QLabel("Acquisition")
        self.costLabel = QLabel("Cost Function")
                                
        self.wMain.addWidget(self.nextBtn, row=0, col=0)
        self.wMain.addWidget(self.costLabel, row=1, col=0)
        self.wMain.addWidget(self.costPlot, row = 2, col = 0, colspan = 2)
        self.wMain.addWidget(self.acqLabel, row=3, col=0)
        self.wMain.addWidget(self.acquisitionPlot, row = 4, col = 0, colspan = 2)
        
        self.wMain.show()
        self.wMain.resize(800,800)
        
    def createButtons(self):  
        self.nextBtn = QtGui.QPushButton('+1')        
        self.nextBtn.clicked.connect(self.nextButton)

    def nextButton(self):
        self.optimizer.step()
        self.optimizer.printDebug()

        # Acquitision
        self.acqCurve.setData(x=self.domainX,y=self.optimizer.acquititionEvaluationAtStep.flatten())
        value, acq = self.optimizer.chosenLocationAtStep
        self.sampleXList.append(value[0])
        self.sampleYList.append(self.costEvaluator.getCost(value)[0])
        self.acqPoint.setData(x=value, y=acq)

        # Costs
        value, cost = self.optimizer.getCurrentStateAndCost()
        self.costPoint.setData(x=value, y=np.array([cost]))
        self.sampleCurve.setData(x=self.sampleXList, y=self.sampleYList)



    def createAcquisitionPlot(self):
        acqPlot = pg.PlotWidget(name='Acquisition')
        acqPlot.setLabel('left', 'Acq')
        acqPlot.setLabel('bottom', 'x')

        self.acqCurve = acqPlot.plot(x=self.domainX, y=self.optimizer.acquititionEvaluationAtStep.flatten())
        value, acq = self.optimizer.chosenLocationAtStep
        self.acqPoint = acqPlot.plot(x=value, y=acq, symbolBrush=(255,0,0), symbolPen='w')

        return acqPlot
        
    def createCostPlot(self):
        costplot = pg.PlotWidget(name='Cost')
        costplot.setLabel('left', 'Cost')
        costplot.setLabel('bottom', 'x')

        domain, costs = self.domain.createFunctionEvaluationOnList(self.costEvaluator.getCost)
        self.costCurve = costplot.plot(x=self.domainX, y=costs.flatten())
        value, cost = self.optimizer.getCurrentStateAndCost()
        self.costPoint = costplot.plot(x=value, y=np.array([cost]), symbolBrush=(255,0,0), symbolPen='w')
        self.sampleCurve = costplot.plot(x=value, y=np.array([cost]), symbolBrush=(0,255,0), symbolPen='w',  pen=(255,0,0, 0))

        return costplot
        
    
