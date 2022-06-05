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
        # self.initialUndoStack.setClean()
        # self.initialUndoStack.cleanIndex()
        # self.initialUndoStack.clear()
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
        # self.initialUndoStack.createUndoAction(self.editAction_undoInitialSolution,"self.printInitialSolution")
        # self.initialUndoStack.createRedoAction(self.editAction_redoInitialSolution,"self.printInitialSolution")
        #  edit buttons
        self.initialButton_clearInitialSolution.clicked.connect(lambda: {self.initialSolution_scene.clear(),
                                                                        self.initialUndoStack.clear()})
        # self.initialUndoStack.createUndoAction(self.initialButton_undo,"self.printInitialSolution")
        # self.initialUndoStack.createRedoAction(self.initialButton_redo,"self.printInitialSolution")
        self.initialButton_undo.pressed.connect(lambda: {self.initialUndoStack.undo(),self.holdButton()})
        self.initialButton_undo.released.connect(lambda : self.mouseReleasedEvent())
        # self.initialButton_undo.released.connect(self.releaseEvent)
        self.initialButton_redo.clicked.connect(self.initialUndoStack.redo)
        # self.initialUndoStack.

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
        
    def holdButton(self):
        self.timer = QTimer()
        self.heldTime = 0
        self.timer.start(100)
        self.timer.timeout.connect(self.timePassedEvent)
        
    def mouseReleasedEvent(self):
        print("release")
        self.timer.stop()
        
    def timePassedEvent(self):
        self.heldTime +=1
        print(self.heldTime)
        if self.heldTime >= 8:
            self.showUndoView()
            self.timer.stop()
            
    def showUndoView(self):
        self.undoView = QUndoView()
        self.undoView.setStack(self.initialUndoStack)
        self.undoView.show()
        
    def specialSignalSlots(self):
        self.communicator = SignalSlotCommunicationManager()
        self.communicator.fileOpened.connect(lambda: {self.printInitialSolution(description="Data")})
        self.initialUndoStack.canUndoChanged.connect(lambda: self.enableInitialUndo(self.initialUndoStack.canUndo()))
        self.initialUndoStack.canRedoChanged.connect(lambda: self.enableInitialRedo(self.initialUndoStack.canRedo()))
        
    def resizeEvent(self, event):
        self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
        QtWidgets.QMainWindow.resizeEvent(self, event)
        
    def enableAfterDataObtained(self):
        # print("enableAfterDataObtained")
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
        # self.printInitialSolution()
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
      

            
        
    def printInitialSolution(self,description="", labels=[],centers=[]): 
        data = self.DataHolder.getInitialData()
        self.initialUndoStack.beginMacro(description)
        self.initialUndoStack.push(InitialSolutionGraph(self.initialSolution_graphicsView,self.initialSolution_scene,"print data",data,labels,centers))
        self.initialUndoStack.endMacro()
        
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
        self.kmWindow.OKButton.clicked.connect(lambda:{self.initialSolutionResults(description="K-Means")})
        
    def affinityPropagation(self):
        # self.stack.push(self.affinityPropagation)
        self.apWindow = AffinityPropagationCalculator()
        self.apWindow.show()
        self.apWindow.OKButton.clicked.connect(lambda:{self.initialSolutionResults(description="Affinity Propagation")})
        
    def meanShift(self):
        # self.stack.push(self.meanShift)
        self.msWindow = meanShiftCalculator()
        self.msWindow.show()
        self.msWindow.OKButton.clicked.connect(lambda:{self.initialSolutionResults(description="Mean-Shift")})
        
    def dbScan(self):
        # self.stack.push(self.dbScan)
        self.dbScanWindow = dbScanCalculator()
        self.dbScanWindow.show()
        self.dbScanWindow.OKButton.clicked.connect(lambda:{self.initialSolutionResults(description="DBSCAN")})
        
    def hClustering(self):
        # self.stack.push(self.hClustering)
        self.hcWindow = hcCalculator()
        self.hcWindow.show()
        self.hcWindow.OKButton.clicked.connect(lambda:{self.initialSolutionResults(description="Hierarchical Clustering")})
        
    def spectralClustering(self):
        # self.stack.push(self.spectralClustering)
        self.scWindow = scCalculator()
        self.scWindow.show()
        self.scWindow.OKButton.clicked.connect(lambda:{self.initialSolutionResults(description="Spectral Clustering")})
        
    
    def initialSolutionResults(self,description):
        self.results_textBrowser.clear()
        labels=self.DataHolder.getLabels()
        print(labels)
        centers=self.DataHolder.getCenters()
        self.printInitialSolution(description,labels,centers)
        self.results_textBrowser.append("Clustering Labels:")
        self.results_textBrowser.append(str(labels))
        self.results_textBrowser.append("\n")
        clusters = self.DataHolder.getClusterIndices()
        for i in range(0,len(clusters)):
            self.results_textBrowser.append("Cluster "+str(i)+"-->"+str(*clusters[i]))
        center_nodes = self.DataHolder.getCenterNodes()
        self.results_textBrowser.append("\nCluster center nodes -->"+str(center_nodes))
        farhest_distances = self.DataHolder.getFarhestHubDistances()
        self.results_textBrowser.append("\n****Farhest hub distances****\n"+str(farhest_distances))
        pair_combinations = self.DataHolder.getPairCombinations()
        self.results_textBrowser.append("\nAll possible pairs: "+str(pair_combinations))
        pair_objectives = self.DataHolder.getPairObjectives()
        self.results_textBrowser.append("\n****Pair objectives****\n"+str(pair_objectives))
        objective_result = self.DataHolder.getObjectiveResult()
        self.results_textBrowser.append("\nObjective function -->"+str(objective_result))
        
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

                  
    
        

        