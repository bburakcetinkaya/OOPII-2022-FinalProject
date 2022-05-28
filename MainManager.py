# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:56:23 2022

@author: Burak Çetinkaya
         151220152110
"""

from UndoRedo import InitialSolutionGraph
from MainWindow import Ui_MainWindow
from PyQt5.QtCore import QObject,pyqtSignal
from DataHolder import DataHolder
from SignalSlotCommunicationManager import SignalSlotCommunicationManager
from dataOperations import KMeansCalculator
from dataOperations import AffinityPropagationCalculator
from dataOperations import meanShiftCalculator
from dataOperations import dbScanCalculator
from dataOperations import hcCalculator
from dataOperations import scCalculator
from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtWidgets import QFileDialog,QUndoStack,QUndoCommand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainManager(QtWidgets.QMainWindow,Ui_MainWindow):  
    
    def __init__(self,parent=None):
        super().__init__(parent)
        QtCore.QObject.__init__(self)
        self.setupUi(self)
        
        # self.createUndoRedo()
        self.undoStack = QUndoStack()
        self.connectSignalSlots()        
        self.dataHolder = DataHolder()
        self.specialSignalSlots() 
        self.initialSolution_scene = QtWidgets.QGraphicsScene(self) 
        
        dummyCmd = InitialSolutionGraph(self.initialSolution_graphicsView, self.initialSolution_scene)
        self.undoStack.push(dummyCmd)

        # self.initialSolution_scene.addWidget(self.initialSolution_canvas)

    def connectSignalSlots(self):
        # file actions
        self.fileAction_openData.triggered.connect(self.openFile)
        self.fileAction_Exit.triggered.connect(lambda: self.close())
        self.fileAction_saveInitialSolution.triggered.connect(lambda : self.saveToFile("save","initial"))
        self.fileAction_exportAsInitialSolution.triggered.connect(lambda :self.saveToFile("saveAs","initial"))        
        # file buttons
        self.fileButton_open.clicked.connect(self.openFile)     
        self.initialButton_saveInitialSolution.clicked.connect(lambda : self.saveToFile("save","initial"))
        self.initialButton_exportAsInitialSolution.clicked.connect(lambda :self.saveToFile("saveAs","initial"))
        
        #  edit actions
        self.editAction_clearInitialSolution.triggered.connect(lambda: self.initialSolution_scene.clear())
        self.editAction_undoInitialSolution.triggered.connect(self.undoStack.undo)
        self.editAction_redoInitialSolution.triggered.connect(self.undoStack.redo)
        #  edit buttons
        self.initialButton_clearInitialSolution.clicked.connect(lambda: self.initialSolution_scene.clear())
        self.initialButton_undo.clicked.connect(self.undoStack.undo)
        self.initialButton_redo.clicked.connect(self.undoStack.redo)
        # self.initial
        
        # clustering actions
        self.clusteringAction_KMeans.triggered.connect(self.kMeans)
        self.clusteringAction_affinityPropagation.triggered.connect(self.affinityPropagation)
        self.clusteringAction_DBSCAN.triggered.connect(self.dbScan)
        self.clusteringAction_hierarchicalClustering.triggered.connect(self.hClustering)
        self.clusteringAction_meanShift.triggered.connect(self.meanShift)
        self.clusteringAction_spectralClustering.triggered.connect(self.spectralClustering)
        # clustering buttons
        self.clusteringButton_KMeans.clicked.connect(self.kMeans)
        self.clusteringButton_affinityPropagation.clicked.connect(self.affinityPropagation)
        self.clusteringButton_hierarchicalClustering.clicked.connect(self.hClustering)
        self.clusteringButton_meanShift.clicked.connect(self.meanShift)
        self.clusteringButton_DBSCAN.clicked.connect(self.dbScan)
        self.clusteringButton_spectralClustering.clicked.connect(self.spectralClustering)
        
    def specialSignalSlots(self):
        self.communicator = SignalSlotCommunicationManager()
        self.communicator.fileOpened.connect(self.printInitialSolution)
        
    # def resizeEvent(self, event):
    #     self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
    #     QtWidgets.QMainWindow.resizeEvent(self, event)
        
    def enableAfterDataObtained(self):
        print("enableAfterDataObtained")
        # file actions
        self.fileAction_saveFinalSolution.setEnabled(True)
        # file buttons
        self.initialButton_saveInitialSolution.setEnabled(True)
        
        # edit actions
        self.editAction_clearInitialSolution.setEnabled(True)
        self.editAction_undoInitialSolution.setEnabled(True)
        self.editAction_redoInitialSolution.setEnabled(True)
        #  edit buttons
        self.initialButton_clearInitialSolution.setEnabled(True)
        self.initialButton_undo.setEnabled(True)
        self.initialButton_redo.setEnabled(True)
        
        # clustering actions
        self.clusteringAction_KMeans.setEnabled(True)
        self.clusteringAction_affinityPropagation.setEnabled(True)
        self.clusteringAction_meanShift.setEnabled(True)
        self.clusteringAction_spectralClustering.setEnabled(True)
        self.clusteringAction_hierarchicalClustering.setEnabled(True)
        self.clusteringAction_DBSCAN.setEnabled(True)
        # clustering buttons
        self.clustering_hLayout.setEnabled(True)
        
    def openFile(self):
        # self.stack.push(self.openFile)
        print("openFile")
        self.filePath = QFileDialog.getOpenFileName(filter = "Data files (*.txt)")[0]
        df = pd.read_csv(self.filePath,sep=" ",header=None)
        df.columns = ["X","Y"]
        data = np.array(df)
        self.dataHolder.setInitialData(data)
        self.communicator.fileOpened.emit()
        self.enableAfterDataObtained()
        # self.undoRedo(self.openFile, [],[],lambda:{print("asdf")})
        # self.printInitialSolution(self,labels=[],centers=[])

    def saveToFile(self,option,solution):  
        self.stack.push(self.saveToFile)
        if solution == "initial":
            data = self.getInitialData()
            if option == "save":
                np.savetxt("initial_solution.txt",data, fmt='%.6f')
            if option == "saveAs":
                filePath = QFileDialog.getSaveFileName(filter = ("TXT(*.txt"))[0]
                np.savetxt(filePath, data,fmt="%.6f") 
                
        if solution == "final":
            data = self.getFinalData()
            if option == "save":                
                np.savetxt("final_solution.txt", data,fmt='%.6f')
            if option == "saveAs":
                filePath = QFileDialog.getSaveFileName(filter = ("TXT(*.txt"))[0]
                np.savetxt(filePath, data,fmt="%.6f")
      

            
        
    def printInitialSolution(self,labels=[],centers=[]): 
        data = self.dataHolder.getInitialData()
        command = InitialSolutionGraph(self.initialSolution_graphicsView,self.initialSolution_scene,data,labels,centers,)
        self.undoStack.push(command)

        # self.iter = 0
        # data = self.dataHolder.getInitialData()
        # command = InitialSolutionUndoRedo(self.initialSolution_graphicsView)
        # self.undoStack.push(command)
        # self.iter +=1
        # # self.stack.push(self.printInitialSolution)                       
        
        # # self.initialSolution_scene = QtWidgets.QGraphicsScene(self) 
        # self.initialSolution_figure = plt.figure(dpi=100)
        # self.initialSolution_canvas = FigureCanvas(self.initialSolution_figure)
        # self.initialSolution_scene = QtWidgets.QGraphicsScene(self) 
                
        # self.initialSolution_figure.clear()
        # ploting = self.initialSolution_figure.add_subplot(111)
        # ploting.scatter(data[:,0], data[:,1],color="k",s=50) 
        # print("lbl:",labels,"centers:",centers)

        # if len(labels):
        #     ploting.scatter(data[:,0], data[:,1],c = labels,s = 50,cmap = 'rainbow')
        #     # print("lbl")
        # if len(centers):
        #     ploting.scatter(centers[:, 0],centers[:, 1],c = "red",s = 50, marker="x",alpha = 1,linewidth=1)
        #     # print("center")
            
        
       
        # # self.initialSolution_scene.update(self.initialSolution_canvas)
        # self.initialSolution_scene.addWidget(self.initialSolution_canvas)
        # # cmd = self.undoRedo.createUndoAction(self.initialSolution_scene,"undo")
        # # print(self.undoRedo.canUndo())
        
        # # print(self.undoRedo.command(1))
        
        # self.initialSolution_graphicsView.setScene(self.initialSolution_scene)   
        # self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
        
    def kMeans(self):
        # self.stack.push(self.kMeans)
        self.kmWindow = KMeansCalculator()
        self.kmWindow.show()
        self.kmWindow.OKButton.clicked.connect(lambda:{self.printInitialSolution(self.kmWindow.getLabels(),
                                                                                 self.kmWindow.getCenters())})
    def affinityPropagation(self):
        # self.stack.push(self.affinityPropagation)
        self.apWindow = AffinityPropagationCalculator()
        self.apWindow.show()
        self.apWindow.OKButton.clicked.connect(lambda:{self.printInitialSolution(self.apWindow.getLabels(),
                                                                                 self.apWindow.getCenters())})
    def meanShift(self):
        # self.stack.push(self.meanShift)
        self.msWindow = meanShiftCalculator()
        self.msWindow.show()
        self.msWindow.OKButton.clicked.connect(lambda:{self.printInitialSolution(self.msWindow.getLabels(),
                                                                                 self.msWindow.getCenters())})
    def dbScan(self):
        # self.stack.push(self.dbScan)
        self.dbScanWindow = dbScanCalculator()
        self.dbScanWindow.show()
        self.dbScanWindow.OKButton.clicked.connect(lambda:{self.printInitialSolution(self.dbScanWindow.getLabels(),
                                                                                     self.dbScanWindow.getCenters())})
    def hClustering(self):
        # self.stack.push(self.hClustering)
        self.hcWindow = hcCalculator()
        self.hcWindow.show()
        self.hcWindow.OKButton.clicked.connect(lambda:{self.printInitialSolution(self.hcWindow.getLabels(),
                                                                                 self.hcWindow.getCenters())})
    def spectralClustering(self):
        # self.stack.push(self.spectralClustering)
        self.scWindow = scCalculator()
        self.scWindow.show()
        self.scWindow.OKButton.clicked.connect(lambda:{self.printInitialSolution(self.scWindow.getLabels(),
                                                                                 self.scWindow.getCenters())})
    
        


    def undo(self):
        print("main undo")
        # self.undoRedo.undo()
        self.undoStack.undo()
        # self.printInitialSolution(self.undoRedo.undo())
        # print(self.undoRedo.undoRedoStack)
        
    def redo(self):
        print("main redo")
        self.undoStack.redo()
        # self.printInitialSolution(self.undoRedo.redo())
        # print(self.undoRedo.undoRedoStack)
         


        

        # self.initialButton_exportAsInitialSolution.setText(_translate("MainWindow", "..."))
        # self.toolButton_14.setText(_translate("MainWindow", "..."))
        # self.toolButton_11.setText(_translate("MainWindow", "..."))
        # self.initialButton_clearInitialSolution.setText(_translate("MainWindow", "..."))
       
        # self.finalButton_saveFinalSolution.setText(_translate("MainWindow", "..."))
        # self.finalButton_exportAsFinalSolution.setText(_translate("MainWindow", "..."))
        # self.toolButton_7.setText(_translate("MainWindow", "..."))
        # self.toolButton_6.setText(_translate("MainWindow", "..."))
        # self.finalButton_clearFinalSolution.setText(_translate("MainWindow", "..."))



        # self.heuristicsButton_hillClimbing.setText(_translate("MainWindow", "..."))
        # self.heuristicsButton_simulatedAnneling.setText(_translate("MainWindow", "..."))



        # self.manualSolution_runButton.setText(_translate("MainWindow", "RUN"))

        # self.menuFile.setTitle(_translate("MainWindow", "File"))
        # self.menuExport_As.setTitle(_translate("MainWindow", "Export As"))
        # self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        # self.menuClear.setTitle(_translate("MainWindow", "Clear"))

        # self.fileAction_openData.setShortcut(_translate("MainWindow", "Ctrl+O"))
       
        
        # self.fileAction_exportAsFinalSolution.setText(_translate("MainWindow", "Final Solution"))
        

        
        # self.editAction_Undo.setText(_translate("MainWindow", "Undo"))
        # self.editAction_Redo.setText(_translate("MainWindow", "Redo"))
        # self.editAction_clearInitialSolution.setText(_translate("MainWindow", "Initial Solution"))
        # self.editAction_clearFinalSolution.setText(_translate("MainWindow", "Final Solution"))
        
        # self.heuristicsAction_hillClimbing.setText(_translate("MainWindow", "Hill Climbing"))
        # self.heuristicsAction_simulatedAnneling.setText(_translate("MainWindow", "Simulated Anneling"))

                  

        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainManager()
    win.show()
    sys.exit(app.exec())
        