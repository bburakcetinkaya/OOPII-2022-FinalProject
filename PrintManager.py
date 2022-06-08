# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 03:22:20 2022

@author: piton
"""

from PyQt5.QtWidgets import QUndoCommand
from SignalSlotCommunicationManager import SignalSlotCommunicationManager
# from MainWindow import Ui_MainWindow
import matplotlib.pyplot as plt
import numpy as np
from DataHolder import DataHolder 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


  
class Solution(QUndoCommand):

    def __init__(self,info,results,view,scene,selection):
        
        
        super(Solution, self).__init__()
        # print("initial solutition called")
        self.__info = info
        self.__results = results
        self.__selection = selection
        self.__scene = scene
        self.__view = view
        self.__figure = plt.figure()
        self.__canvas = FigureCanvas(self.__figure)

        self.__redoFlag = True
        self.__undoFlag = True
        self.DataHolder = DataHolder()
        # print("init")
        
        self.communicator = SignalSlotCommunicationManager()
        self.communicator.initialClear.connect(lambda: self.__initScene.clear())
        self.communicator.finalClear.connect(lambda: self.__finalScene.clear())
        # self.communicator.undoEvent.connect(self.printGraph)
        # self.communicator.redoEvent.connect(self.printGraph)
        # self.printGraph()
    def printResults(self,selection):        
         self.__info.clear()
 
         if selection == "INITIAL":
              self.__centers=self.DataHolder.getCenters()
              self.__results.append("Clustering Labels:")
              self.__labels=self.DataHolder.getLabels()  
              self.__results.append(str(self.__labels))
              self.__results.append("\n")
              clusters = self.DataHolder.getClusterNodes()
              for i in range(0,len(clusters)):
                  self.__results.append("Cluster "+str(i)+"-->"+str(clusters[i]))
              center_nodes = self.DataHolder.getCenterNodes()
              self.__results.append("\nCluster center nodes -->"+str(center_nodes))
              farhest_distances = self.DataHolder.getFarhestHubDistances()
              self.__results.append("\n****Farhest hub distances****\n"+str(farhest_distances))
              pair_combinations = self.DataHolder.getPairCombinations()
              self.__results.append("\nAll possible pairs: "+str(pair_combinations))
              pair_objectives = self.DataHolder.getPairObjectives()
              self.__results.append("\n****Pair objectives****\n"+str(pair_objectives))
              objective_result = self.DataHolder.getObjectiveResult()
              self.__results.append("\nObjective function -->"+str(objective_result))
              
              for i in range(0,len(clusters)):
                 self.__info.append("Cluster "+str(i)+"-->"+str(clusters[i]))
              self.__info.append("\nCluster center nodes -->"+str(center_nodes))
              self.__info.append("\nObjective function -->"+str(objective_result))

         if selection == "FINAL":
             clusters = self.DataHolder.getBestClusterNodes()
             for i in range(0,len(clusters)):
                self.__results.append("Cluster "+str(i)+"-->"+str(clusters[i]))
             center_nodes = self.DataHolder.getBestCenterNodes()
             self.__results.append("\nCluster center nodes -->"+str(center_nodes))
             farhest_distances = self.DataHolder.getBestFarhestHubDistances()
             self.__results.append("\n****Farhest hub distances****\n"+str(farhest_distances))
             pair_combinations = self.DataHolder.getBestPairCombinations()
             self.__results.append("\nAll possible pairs: "+str(pair_combinations))
             pair_objectives = self.DataHolder.getBestPairObjectives()
             self.__results.append("\n****Pair objectives****\n"+str(pair_objectives))
             objective_result = self.DataHolder.getBestObjectiveResult()
             self.__results.append("\nObjective function -->"+str(objective_result))
             total_iterations = self.DataHolder.getTotalIterations()
             self.__results.append("\nTotal iterations -->"+str(total_iterations))
             result_iterations = self.DataHolder.getResultIterationsNumber()
             self.__results.append("\nResult found iteration -->"+str(result_iterations))
             execution_time = self.DataHolder.getExecutionTime()
             self.__results.append("\nExecution time(s) -->"+str(execution_time))
             
             for i in range(0,len(clusters)):
                self.__info.append("Cluster "+str(i)+"-->"+str(clusters[i]))
             self.__info.append("\nCluster center nodes -->"+str(center_nodes))
             self.__info.append("\nObjective function -->"+str(objective_result))
             self.__info.append("\nObjective function -->"+str(objective_result))      
             self.__info.append("\nTotal iterations -->"+str(total_iterations))
             self.__info.append("\nResult found iteration -->"+str(result_iterations))
             self.__info.append("\nExecution time(s) -->"+str(execution_time))
                 
         if selection == "DATA":
             self.__results.append("Data read")
             self.__info.append("Data read")
         
    def printInitialGraph(self):
        self.__scene.clear()  
        self.__data = self.DataHolder.getInitialData()
        self.__labels=self.DataHolder.getLabels()  
        self.__centers = self.DataHolder.getCenters()
        self.__figure = plt.figure()
        self.__canvas = FigureCanvas(self.__figure)  
        self.__clusterNodes = self.DataHolder.getClusterNodes()
        ploting = self.__figure.add_subplot(111)
        
        if len(self.__data):
            for i in range(len(self.__data)):
                ploting.scatter(self.__data[i,0], self.__data[i,1],c="k")
                plt.annotate(str(i), (self.__data[i,0], self.__data[i,1]))
        
        if len(self.__labels):
            ploting.scatter(self.__data[:,0], self.__data[:,1],c = self.__labels,s = 50,cmap = 'rainbow')
          
        if len(self.__centers):
            ploting.scatter(np.array(self.__centers)[:, 0],np.array(self.__centers)[:, 1],c = "red",s = 100, marker="x",alpha = 1,linewidth=1)

                
        self.__scene.addWidget(self.__canvas)
        self.__view.setScene(self.__scene)   
        self.__view.fitInView(self.__scene.sceneRect())
    def printFinalGraph(self):
        self.__scene.clear()
        self.__data = self.DataHolder.getInitialData()
        self.__labels=self.DataHolder.getLabels()  
        self.__centers = self.DataHolder.getBestCenters()
          
        ploting = self.__figure.add_subplot(111)
        
        if len(self.__data):
            ploting.scatter(self.__data[:,0], self.__data[:,1],color="k",s=50) 
 
        # if len(self.__labels):
        #     ploting.scatter(self.__data[:,0], self.__data[:,1],c = self.__labels,s = 50,cmap = 'rainbow')
        # color = np.arange(0, self.DataHolder.getNumberOfClusters(), 1, dtype=int)
        # print(color)
        for i in range(0,self.DataHolder.getNumberOfClusters()):
            ploting.scatter(self.__data[:,0], self.__data[:,1])#,c=color)    
          
        if len(self.__centers):
            ploting.scatter(np.array(self.__centers)[:, 0],np.array(self.__centers)[:, 1],c = "red",s = 100, marker="x",alpha = 1,linewidth=1)

        self.__scene.addWidget(self.__canvas)
        self.__scene.setScene(self.__scene)   
        self.__scene.fitInView(self.__scene.sceneRect())
            
    def undo(self): 
        # self.communicator.undoEvent.emit()
        # self.printResults()
        print("undo")
        # if self.__selection == "INITIAL":
        #     self.printInitialGraph()
        # if self.__selection == "FINAL":
        #     self.printFinalGraph()
        
    def redo(self):
        # self.communicator.redoEvent.emit()
        # self.printResults()
        if self.__selection == "INITIAL" or self.__selection == "DATA":
            self.printInitialGraph()
            
        if self.__selection == "FINAL":
            self.printFinalGraph()
        
        self.printResults(self.__selection)



  