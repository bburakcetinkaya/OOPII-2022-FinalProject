# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:56:23 2022

@author: Burak Çetinkaya
         151220152110
"""

from PrintManager import Solution
from MainWindow import Ui_MainWindow

from DataHolder import DataHolder
from SignalSlotCommunicationManager import SignalSlotCommunicationManager
from dataOperations import KMeansCalculator
from dataOperations import AffinityPropagationCalculator
from dataOperations import meanShiftCalculator
from dataOperations import dbScanCalculator
from dataOperations import hcCalculator
from dataOperations import scCalculator

from dataOperations import SimulatedAnneling,HillClimbing

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
        self.initialUndoStack = QUndoStack()
        self.finalUndoStack = QUndoStack()
        self.initialUndoStack.clear()
        self.finalUndoStack.clear()
        self.initialButton_undo.setEnabled(False)
        self.initialButton_redo.setEnabled(False)
        
        self.communicator = SignalSlotCommunicationManager()
        self.finalSolution_scene = QtWidgets.QGraphicsScene() 
        self.initialSolution_scene = QtWidgets.QGraphicsScene() 
        self.connectSignalSlots()        
        self.DataHolder = DataHolder()
        self.specialSignalSlots()
        
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
        self.editAction_clearFinalSolution.triggered.connect(lambda: {self.initialSolution_scene.clear(),
                                                                      self.initialUndoStack.clear()})

        #  edit buttons
        self.initialButton_clearInitialSolution.clicked.connect(lambda: {self.initialSolution_scene.clear(),
                                                                        self.initialUndoStack.clear()})
        self.finalButton_clearFinalSolution.clicked.connect(lambda: {self.finalSolution_scene.clear(),         
                                                                     self.finalUndoStack.clear()})

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
        
        #heuristics actions
        self.heuristicsAction_hillClimbing.triggered.connect(self.hillClimbing)
        self.heuristicsAction_simulatedAnneling.triggered.connect(self.simulatedAnneling)
    def hillClimbing(self):
        HillClimbing(100000)
        self.SolutionStackControl(selection="FINAL",description="Hill Climbing")
        self.enableAfterHeuristics()
        
    def simulatedAnneling(self):
        SimulatedAnneling(100000)
        self.SolutionStackControl(selection="FINAL",description="Simulated Anneling")
        self.enableAfterHeuristics()
        
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
        self.communicator.fileOpened.connect(lambda: {self.SolutionStackControl(selection = "DATA" , description="Data read")})
        self.initialUndoStack.canUndoChanged.connect(lambda: self.enableInitialUndo(self.initialUndoStack.canUndo()))
        self.initialUndoStack.canRedoChanged.connect(lambda: self.enableInitialRedo(self.initialUndoStack.canRedo()))
        self.finalUndoStack.canUndoChanged.connect(lambda: self.enableFinalUndo(self.finalUndoStack.canUndo()))
        self.finalUndoStack.canRedoChanged.connect(lambda: self.enableFinalRedo(self.finalUndoStack.canRedo()))
        
    def resizeEvent(self, event):
        self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
        self.finalSolution_graphicsView.fitInView(self.finalSolution_scene.sceneRect())
        QtWidgets.QMainWindow.resizeEvent(self, event)
    
    def enableAfterHeuristics(self):
        self.finalSolution_groupBox.setEnabled(True)
        self.finalSolution_hLayout.setEnabled(True)
        self.finalButton_clearFinalSolution.setEnabled(True)
        self.finalButton_exportAsFinalSolution.setEnabled(True)
        self.finalButton_redo.setEnabled(True)
        self.finalButton_undo.setEnabled(True)
        self.editAction_redoFinalSolution.setEnabled(True)
        self.editAction_clearFinalSolution.setEnabled(True)
        self.editAction_undoFinalSolution.setEnabled(True)
        self.fileAction_exportAsFinalSolution.setEnabled(True)
        
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
        
        self.DataHolder.setInitialData(np.array(0))
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
      

            
        
    def SolutionStackControl(self,selection,description=""): 
        if selection == "INITIAL" or selection == "DATA":
            self.initialUndoStack.beginMacro(description)
            pushCommand = Solution(self.infoPanel_textEdit,self.results_textBrowser,self.initialSolution_graphicsView,self.initialSolution_scene,selection)
            self.initialUndoStack.push(pushCommand)
            self.initialUndoStack.endMacro()
        if selection == "FINAL":
            self.finalUndoStack.beginMacro(description)
            pushCommand = Solution(self.infoPanel_textEdit,self.results_textBrowser,self.finalSolution_graphicsView,self.finalSolution_scene,selection)
            self.finalUndoStack.push(pushCommand)
            self.finalUndoStack.endMacro()
        
            
    def kMeans(self):
        # self.stack.push(self.kMeans)
        self.kmWindow = KMeansCalculator()
        self.kmWindow.show()
        self.kmWindow.OKButton.clicked.connect(lambda:{self.SolutionStackControl(selection="INITIAL",description="K-Means")})
        self.enableAfterClustering()
    def affinityPropagation(self):
        # self.stack.push(self.affinityPropagation)
        self.apWindow = AffinityPropagationCalculator()
        self.apWindow.show()
        self.apWindow.OKButton.clicked.connect(lambda:{self.SolutionStackControl(selection="INITIAL",description="Affinity Propagation")})
        self.enableAfterClustering()
    def meanShift(self):
        self.msWindow = meanShiftCalculator()
        self.msWindow.show()
        self.msWindow.OKButton.clicked.connect(lambda:{self.SolutionStackControl(selection="INITIAL",description="Mean-Shift")})
        self.enableAfterClustering()
    def dbScan(self):
        self.dbScanWindow = dbScanCalculator()
        self.dbScanWindow.show()
        self.dbScanWindow.OKButton.clicked.connect(lambda:{self.SolutionStackControl(dselection="INITIAL",escription="DBSCAN")})
        self.enableAfterClustering()
    def hClustering(self):
        self.hcWindow = hcCalculator()
        self.hcWindow.show()
        self.hcWindow.OKButton.clicked.connect(lambda:{self.SolutionStackControl(selection="INITIAL",description="Hierarchical Clustering")})
        self.enableAfterClustering()
    def spectralClustering(self):
        self.scWindow = scCalculator()
        self.scWindow.show()
        self.scWindow.OKButton.clicked.connect(lambda:{self.SolutionStackControl(selection="INITIAL",description="Spectral Clustering")})
        self.enableAfterClustering()
    def enableInitialUndo(self,selection):
         self.initialButton_undo.setEnabled(selection)
         self.editAction_undoInitialSolution.setEnabled(selection)
         
    def enableInitialRedo(self,selection):        
        self.initialButton_redo.setEnabled(selection)        
        self.editAction_redoInitialSolution.setEnabled(selection)
        
    def enableFinalUndo(self,selection):
         self.finalButton_undo.setEnabled(selection)
         self.editAction_undoFinalSolution.setEnabled(selection)
         
    def enableFinalRedo(self,selection):       
        self.finalButton_redo.setEnabled(selection)
        self.editAction_redoFinalSolution.setEnabled(selection)

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

                  
    
        

        