from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QUndoCommand

# from MainWindow import Ui_MainWindow

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from enum import Enum
class Vars(Enum):
        data = 0
        labels = 1
        centers = 2
        
class InitialSolutionGraph(QUndoCommand):  # this is gonna a lot  tougher

    def __init__(self,initialSolution_graphicsView,initialSolution_scene,data=[],labels=[],centers=[]):  # I've got an Idea
        """
        Method defines Cut function for table Widget
        Args:
             model (QModel*):
             buffer_copy (list [Row,Column, QIndex][Value]): data_copy
             cut (str) ''
             description (str) description of the Process
             tableAddress (QWidget*) TableWidget Reference
        """
        super(InitialSolutionGraph, self).__init__()
        # print("initial solutition called")
        
        self.initialSolution_graphicsView = initialSolution_graphicsView
        self.initialSolution_scene = initialSolution_scene
        self.__data = data
        self.__labels = labels
        self.__centers = centers
        self.printGraph()
        

    def printGraph(self):
        
        if not len(self.__data):
            return

        self.initialSolution_figure = plt.figure(dpi=100)     
        self.initialSolution_canvas = FigureCanvas(self.initialSolution_figure)           
        self.initialSolution_figure.clear()
        
        ploting = self.initialSolution_figure.add_subplot(111)
        

        if len(self.__data):
            ploting.scatter(self.__data [:,0], self.__data [:,1],color="k",s=50) 
        if len(self.__labels) and len(self.__centers):
            ploting.scatter(self.__data[:,0], self.__data[:,1],c = self.__labels,s = 50,cmap = 'rainbow')
            ploting.scatter(self.__centers[:, 0],self.__centers[:, 1],c = "red",s = 50, marker="x",alpha = 1,linewidth=1)    

            
            
        self.initialSolution_scene.addWidget(self.initialSolution_canvas)            
        self.initialSolution_graphicsView.setScene(self.initialSolution_scene)   
        self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
    
            
    def undo(self): 
        # self.printGraph()
        # print("lbl:",self.__labels)
        self.printGraph()
        print("stack undo")
        print()

    def redo(self):
        self.printGraph()
        # print("lbl:",self.__labels)
        print("stack redo")
        # self.printGraph()
        print()