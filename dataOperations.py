# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:19:55 2022

@author: piton
"""
from DataHolder import DataHolder
from PyQt5 import QtWidgets,QtGui
from KMeansWindow import Ui_kMeansWindow
import numpy as np
from sklearn.cluster import KMeans
from SignalSlotCommunicationManager import SignalSlotCommunicationManager

class KMeansCalculator(QtWidgets.QMainWindow,Ui_kMeansWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
       
        self.setupUi(self)
        # self.show()
        # print(initialData)
        self.connectSignalSlots()
        self.DataHolder = DataHolder()
        
    def connectSignalSlots(self):
        self.OKButton.clicked.connect(self.calculate)
        self.resetButton.clicked.connect(self.resetParameters)

    # def getInitialData(self):
    #     return self.__initialData
    
    def calculate(self):
        self.n_clusters = int(self.n_clusters_linedit.text())
        self.init = str(self.init_comboBox.currentText())
        self.n_init = int(self.n_init_lineEdit.text())
        self.max_iter = int(self.max_iter_lineEdit.text())
        self.tol = float(self.tol_lineEdit.text())
        self.algorithm = str(self.algorithm_comboBox.currentText())
        # print(self.n_clusters,self.init,self.n_init,self.max_iter,self.tol,self.algorithm)
        data = self.DataHolder.getInitialData()
        self.kmeans = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, 
                             max_iter=self.max_iter, tol=self.tol, algorithm=self.algorithm)
        
        km = self.kmeans.fit(data)
        self.setLabels(km.labels_)
        # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
        # cluster = np.hstack((ind,labels[:,None]))
        # cluster = cluster[cluster[:, -1].argsort()]
        # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
        self.setCenters(km.cluster_centers_)
        
        
    def setCenters(self,centers):
        self.kMeans_centers = centers
        # print("setcenter",self.kMeans_centers)
    def getCenters(self):
        # print("getcenter",self.kMeans_centers)
        return self.kMeans_centers
    
    def setLabels(self,labels):
        self.kMeans_labels = labels
        # print("setlbl",self.kMeans_labels)
    def getLabels(self):
        # print("getlbl",self.kMeans_labels)
        return self.kMeans_labels
        

        # print(self.kmeans.cluster_centers_)
        print("OK")
    def resetParameters(self):
        self.n_clusters_linedit.setText("8")
        self.init_comboBox.setCurrentIndex(0)
        self.n_init_lineEdit.setText("10")
        self.max_iter_lineEdit.setText("300")
        self.tol_lineEdit.setText("0.0001")
        self.algorithm_comboBox.setCurrentIndex(0)

        