from PyQt5.QtWidgets import QUndoCommand
from PyQt5 import QtWidgets
from SignalSlotCommunicationManager import SignalSlotCommunicationManager
# from MainWindow import Ui_MainWindow
import matplotlib.pyplot as plt
import numpy as np
from DataHolder import DataHolder 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from abc import ABCMeta,abstractstaticmethod

  
class InitialSolutionGraph(QUndoCommand):  # this is gonna a lot  tougher

    def __init__(self,results_textBrowser,initialSolution_graphicsView):
        
        
        super(InitialSolutionGraph, self).__init__()
        # print("initial solutition called")
        
        self.results_textBrowser = results_textBrowser
        self.initialSolution_scene = QtWidgets.QGraphicsScene() 
        self.initialSolution_graphicsView = initialSolution_graphicsView

        self.__redoFlag = True
        self.__undoFlag = True
        self.DataHolder = DataHolder()
        # print("init")
        
        # self.communicator = SignalSlotCommunicationManager()
        # self.communicator.undoEvent.connect(self.printGraph)
        # self.communicator.redoEvent.connect(self.printGraph)
        # self.printGraph()
        
    def printResults(self):        
         self.results_textBrowser.clear()
         self.__labels=self.DataHolder.getLabels()         
         # print(self.__labels)
         # self.__centers=self.DataHolder.getCenters()
         # self.results_textBrowser.append("Clustering Labels:")
         # self.results_textBrowser.append(str(self.__labels))
         # self.results_textBrowser.append("\n")
         # clusters = self.DataHolder.getClusterIndices()
         # for i in range(0,len(clusters)):
         #     self.results_textBrowser.append("Cluster "+str(i)+"-->"+str(*clusters[i]))
         # center_nodes = self.DataHolder.getCenterNodes()
         # self.results_textBrowser.append("\nCluster center nodes -->"+str(center_nodes))
         # farhest_distances = self.DataHolder.getFarhestHubDistances()
         # self.results_textBrowser.append("\n****Farhest hub distances****\n"+str(farhest_distances))
         # pair_combinations = self.DataHolder.getPairCombinations()
         # self.results_textBrowser.append("\nAll possible pairs: "+str(pair_combinations))
         # pair_objectives = self.DataHolder.getPairObjectives()
         # self.results_textBrowser.append("\n****Pair objectives****\n"+str(pair_objectives))
         # objective_result = self.DataHolder.getObjectiveResult()
         # self.results_textBrowser.append("\nObjective function -->"+str(objective_result))
         
    def printGraph(self):
        
        self.__data = self.DataHolder.getInitialData()
        self.__labels=self.DataHolder.getLabels()  
        self.__centers = self.DataHolder.getCenters()
        self.initialSolution_figure = plt.figure()
        self.initialSolution_canvas = FigureCanvas(self.initialSolution_figure)  
        ploting = self.initialSolution_figure.add_subplot(111)
        
        if len(self.__data):
            ploting.scatter(self.__data[:,0], self.__data[:,1],color="k",s=50) 
 
        if len(self.__labels):
            ploting.scatter(self.__data[:,0], self.__data[:,1],c = self.__labels,s = 50,cmap = 'rainbow')
          
        if len(self.__centers):
            ploting.scatter(np.array(self.__centers)[:, 0],np.array(self.__centers)[:, 1],c = "red",s = 100, marker="x",alpha = 1,linewidth=1)

        self.initialSolution_scene.addWidget(self.initialSolution_canvas)
        self.initialSolution_graphicsView.setScene(self.initialSolution_scene)   
        self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
            
    def undo(self): 
        # self.communicator.undoEvent.emit()
        # self.printResults()
        self.printGraph()
        
    def redo(self):
        # self.communicator.redoEvent.emit()
        # self.printResults()
        self.printGraph()



  