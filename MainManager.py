# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:56:23 2022

@author: Burak Çetinkaya
         151220152110
"""


from MainWindow import Ui_MainWindow
from DataHolder import DataHolder
from SignalSlotCommunicationManager import SignalSlotCommunicationManager
from dataOperations import KMeansCalculator
from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtWidgets import QFileDialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainManager(QtWidgets.QMainWindow,QtCore.QObject,Ui_MainWindow):    
    def __init__(self,parent=None):
        super().__init__(parent)
        QtCore.QObject.__init__(self)
        self.setupUi(self)
        self.connectSignalSlots()
        self.connectManagersSignalSlots() 
        self.dataHolder = DataHolder()

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
        #  edit buttons
        self.initialButton_clearInitialSolution.clicked.connect(lambda: self.initialSolution_scene.clear())
        
        # clustering actions
        self.clusteringAction_KMeans.triggered.connect(self.kMeans)
        # clustering buttons
        self.clusteringButton_KMeans.clicked.connect(self.kMeans)
        
    def connectManagersSignalSlots(self):
        self.communicator = SignalSlotCommunicationManager()
        self.communicator.fileOpened.connect(self.printInitialSolution)
        # self.communicator.kmeansOKClicked.connect(self.kmeansPrint)
        
    def enableAfterDataObtained(self):
        print("enableAfterDataObtained")
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
        print("openFile")
        self.filePath = QFileDialog.getOpenFileName(filter = "Data files (*.txt)")[0]
        df = pd.read_csv(self.filePath,sep=" ",header=None)
        df.columns = ["X","Y"]
        data = np.array(df)
        self.dataHolder.setInitialData(data)
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
      

            
        
    def printInitialSolution(self,labels=[],centers=[]):        
        self.initialSolution_scene = QtWidgets.QGraphicsScene(self)                
        
        data = self.dataHolder.getInitialData()
        self.figure = plt.figure()
        self.figure.clear()
        ploting = self.figure.add_subplot(111)
        ploting.scatter(data[:,0], data[:,1],color="k",s=50) 
        # print("lbl:",labels,"centers:",centers)

        if len(labels):
            ploting.scatter(data[:,0], data[:,1],c = labels,s = 50,cmap = 'rainbow')
        if len(centers):
            ploting.scatter(centers[:, 0],centers[:, 1],c = "red",s = 50, marker="x",alpha = 1,linewidth=1)
            
        self.initialSolution_canvas = FigureCanvas(self.figure)
        self.initialSolution_scene.addWidget(self.initialSolution_canvas)
        
        self.initialSolution_graphicsView.setScene(self.initialSolution_scene)   
        self.initialSolution_graphicsView.fitInView(self.initialSolution_scene.sceneRect())
        
    def kMeans(self):
        self.kmWindow = KMeansCalculator()
        self.kmWindow.show()
        self.kmWindow.OKButton.clicked.connect(lambda:{self.printInitialSolution(self.kmWindow.getLabels(),
                                                                                 self.kmWindow.getCenters())})
        


        
        


        


        

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
        