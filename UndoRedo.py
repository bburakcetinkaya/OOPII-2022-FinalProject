from PyQt5.QtWidgets import QUndoCommand
from SignalSlotCommunicationManager import SignalSlotCommunicationManager
# from MainWindow import Ui_MainWindow
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    
class InitialSolutionGraph(QUndoCommand):  # this is gonna a lot  tougher

    def __init__(self,initialSolution_graphicsView,initialSolution_scene,data=[],labels=[],centers=[]):
        """

        """
        super(InitialSolutionGraph, self).__init__()
        # print("initial solutition called")
        
        self.initialSolution_graphicsView = initialSolution_graphicsView
        self.initialSolution_scene = initialSolution_scene
        self.__data = data
        self.__labels = labels
        self.__centers = centers  
        self.communicator = SignalSlotCommunicationManager()
        self.communicator.undoEvent.connect(self.printGraph)
        self.communicator.redoEvent.connect(self.printGraph)
    def printGraph(self):
        # print("id = ",QUndoCommand.id(self))
        self.initialSolution_figure = plt.figure()
        self.initialSolution_canvas = FigureCanvas(self.initialSolution_figure)
                
        self.initialSolution_figure.clear()
        ploting = self.initialSolution_figure.add_subplot(111)
        if len(self.__data):
            ploting.scatter(self.__data[:,0], self.__data[:,1],color="k",s=50) 
            print("data")

        if len(self.__labels):
            ploting.scatter(self.__data[:,0], self.__data[:,1],c = self.__labels,s = 50,cmap = 'rainbow')
            print("lbl")
        if len(self.__centers):
            ploting.scatter(self.__centers[:, 0],self.__centers[:, 1],c = "red",s = 50, marker="x",alpha = 1,linewidth=1)
            print("center")
        self.initialSolution_scene.addWidget(self.initialSolution_canvas)
        self.initialSolution_graphicsView.setScene(self.initialSolution_scene)   
        self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
            
    def undo(self): 

        self.communicator.undoEvent.emit()
    def redo(self):

        self.communicator.redoEvent.emit()


  