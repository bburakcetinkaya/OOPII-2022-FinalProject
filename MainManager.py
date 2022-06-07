# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:56:23 2022

@author: Burak Çetinkaya
         151220152110
"""

from UndoRedo import InitialSolutionGraph
from MainWindow import Ui_MainWindow

from DataHolder import DataHolder
from SignalSlotCommunicationManager import SignalSlotCommunicationManager
from dataOperations import KMeansCalculator
from dataOperations import AffinityPropagationCalculator
from dataOperations import meanShiftCalculator
from dataOperations import dbScanCalculator
from dataOperations import hcCalculator
from dataOperations import scCalculator

from Algorithms import Algorithms

from PyQt5 import QtWidgets,QtCore
from PyQt5.QtWidgets import QFileDialog,QUndoStack,QUndoView
from PyQt5.QtCore import QTimer

import numpy as np
import pandas as pd



class MainManager(QtWidgets.QMainWindow,Ui_MainWindow):  
    
    def __init__(self,parent=None):
        super().__init__(parent)
        QtCore.QObject.__init__(self)
        self.setupUi(self)
        
        # self.createUndoRedo()
        self.initialUndoStack = QUndoStack(self)
        # self.initialUndoStack.setClean()
        # self.initialUndoStack.cleanIndex()
        # self.initialUndoStack.clear()
        self.initialButton_undo.setEnabled(False)
        self.initialButton_redo.setEnabled(False)
        
        self.connectSignalSlots()        
        self.DataHolder = DataHolder()
        self.specialSignalSlots() 
        self.initialSolution_scene = QtWidgets.QGraphicsScene(self) 
        # self.undoRedoFlag = True
        # self.initialUndoStack.push(InitialSolutionGraph(self.initialSolution_graphicsView, self.initialSolution_scene))
    

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
        self.editAction_clearInitialSolution.triggered.connect(lambda: {self.initialSolution_scene.clear(),
                                                                        self.initialUndoStack.clear()})
        self.editAction_undoInitialSolution.triggered.connect(self.initialUndoStack.undo)
        self.editAction_redoInitialSolution.triggered.connect(self.initialUndoStack.redo)

        #  edit buttons
        self.initialButton_clearInitialSolution.clicked.connect(lambda: {self.initialSolution_scene.clear(),
                                                                        self.initialUndoStack.clear()})

        self.initialButton_undo.pressed.connect(lambda: {self.initialUndoStack.undo(),self.holdButton()})
        self.initialButton_undo.released.connect(lambda : self.mouseReleasedEvent())

        self.initialButton_redo.clicked.connect(self.initialUndoStack.redo)

    
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
        
        #heuristics buttons
        self.heuristicsButton_hillClimbing.clicked.connect(self.hillClimbing)
        self.heuristicsButton_simulatedAnneling.clicked.connect(self.simulatedAnneling)
    def hillClimbing(self):
        algorithm = Algorithms()
        self.intialSolutionStackControl("Hill Climbing")
        
    def simulatedAnneling(self):
        print("hello")
    def holdButton(self):
        self.timer = QTimer()
        self.heldTime = 0
        self.timer.start(100)
        self.timer.timeout.connect(self.timePassedEvent)
        
    def mouseReleasedEvent(self):

        self.timer.stop()
        
    def timePassedEvent(self):
        self.heldTime +=1
        if self.heldTime >= 8:
            self.showUndoView()
            self.timer.stop()
            
    def showUndoView(self):
        self.undoView = QUndoView()
        self.undoView.setStack(self.initialUndoStack)
        self.undoView.show()
        
    def specialSignalSlots(self):
        self.communicator = SignalSlotCommunicationManager()
        self.communicator.fileOpened.connect(lambda: {self.intialSolutionStackControl(description="Data")})
        self.initialUndoStack.canUndoChanged.connect(lambda: self.enableInitialUndo(self.initialUndoStack.canUndo()))
        self.initialUndoStack.canRedoChanged.connect(lambda: self.enableInitialRedo(self.initialUndoStack.canRedo()))
        
    def resizeEvent(self, event):
        self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
        QtWidgets.QMainWindow.resizeEvent(self, event)
    
    def enableAfterClustering(self):
        self.heuristicsAction_hillClimbing.setEnabled(True)
        self.heuristicsAction_simulatedAnneling.setEnabled(True)
        self.heuristicsButton_hillClimbing.setEnabled(True)
        self.heuristicsButton_simulatedAnneling.setEnabled(True)                                                   
        self.heuristics_hLayout.setEnabled(True)
    def enableAfterDataObtained(self):
        # file actions
        self.fileAction_saveFinalSolution.setEnabled(True)
        # file buttons
        self.initialButton_saveInitialSolution.setEnabled(True)
        
        # edit actions
        self.editAction_clearInitialSolution.setEnabled(True)

        #  edit buttons
        self.initialButton_clearInitialSolution.setEnabled(True)

        
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

        self.filePath = QFileDialog.getOpenFileName(filter = "Data files (*.txt)")[0]
        df = pd.read_csv(self.filePath,sep=" ",header=None)
        df.columns = ["X","Y"]
        data = np.array(df)
        self.DataHolder.setInitialData(data)
        self.communicator.fileOpened.emit()
        self.enableAfterDataObtained()

    def saveToFile(self,option,solution):  

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
      

            
        
    def intialSolutionStackControl(self,description=""): 
        self.initialUndoStack.beginMacro(description)
        pushCommand = InitialSolutionGraph(self.results_textBrowser,self.initialSolution_graphicsView)
        self.initialUndoStack.push(pushCommand)
        self.initialUndoStack.endMacro()
        
    def kMeans(self):
        # self.stack.push(self.kMeans)
        self.kmWindow = KMeansCalculator()
        self.kmWindow.show()
        self.kmWindow.OKButton.clicked.connect(lambda:{self.intialSolutionStackControl(description="K-Means")})
        self.enableAfterClustering()
    def affinityPropagation(self):
        # self.stack.push(self.affinityPropagation)
        self.apWindow = AffinityPropagationCalculator()
        self.apWindow.show()
        self.apWindow.OKButton.clicked.connect(lambda:{self.intialSolutionStackControl(description="Affinity Propagation")})
        self.enableAfterClustering()
    def meanShift(self):
        self.msWindow = meanShiftCalculator()
        self.msWindow.show()
        self.msWindow.OKButton.clicked.connect(lambda:{self.intialSolutionStackControl(description="Mean-Shift")})
        self.enableAfterClustering()
    def dbScan(self):
        self.dbScanWindow = dbScanCalculator()
        self.dbScanWindow.show()
        self.dbScanWindow.OKButton.clicked.connect(lambda:{self.intialSolutionStackControl(description="DBSCAN")})
        self.enableAfterClustering()
    def hClustering(self):
        self.hcWindow = hcCalculator()
        self.hcWindow.show()
        self.hcWindow.OKButton.clicked.connect(lambda:{self.intialSolutionStackControl(description="Hierarchical Clustering")})
        self.enableAfterClustering()
    def spectralClustering(self):
        self.scWindow = scCalculator()
        self.scWindow.show()
        self.scWindow.OKButton.clicked.connect(lambda:{self.intialSolutionStackControl(description="Spectral Clustering")})
        self.enableAfterClustering()
    def enableInitialUndo(self,selection):
         self.initialButton_undo.setEnabled(selection)
         self.editAction_undoInitialSolution.setEnabled(selection)
         
    def enableInitialRedo(self,selection):        
        self.initialButton_redo.setEnabled(selection)        
        self.editAction_redoInitialSolution.setEnabled(selection)

        

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

                  
    
        

        