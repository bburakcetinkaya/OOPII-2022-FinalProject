# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:19:55 2022

@author: piton
"""
from DataHolder import DataHolder
from PyQt5 import QtWidgets
from KMeansWindow import Ui_kMeansWindow
from AffinityPropagationWindow import Ui_apWindow
from DBSCANWindow import Ui_dbScanWindow
from HierarchicalClusteringWindow import Ui_hcWindow
from MeanShiftWindow import Ui_msWindow
from SpectralClusteringWindow import Ui_scWindow

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering


from abc import ABCMeta, abstractmethod

class AbstractCalculator(QtWidgets.QMainWindow):
    __metacalss = ABCMeta
    @abstractmethod
    def connectSignalSlots(self):
        #  connect signals and slots 
        pass
    @abstractmethod
    def calculate(self):
        # makes necessary calculations for clustering
        pass
    @abstractmethod
    def resetParameters(self):
        # resets the parameters in the ui to their defaults
        pass

class KMeansCalculator(AbstractCalculator,Ui_kMeansWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
       
        self.setupUi(self)
        # self.show()
        # print(initialData)
        self.connectSignalSlots()
        self.DataHolder = DataHolder()
        self.kMeans_centers = []
        self.kMeans_labels = []
        
    def connectSignalSlots(self):
        self.OKButton.clicked.connect(self.calculate)
        self.resetButton.clicked.connect(self.resetParameters)

    def calculate(self):
        self.n_clusters = int(self.n_clusters_linedit.text())
        self.init = str(self.init_comboBox.currentText())
        self.n_init = int(self.n_init_lineEdit.text())
        self.max_iter = int(self.max_iter_lineEdit.text())
        self.tol = float(self.tol_lineEdit.text())
        self.algorithm = str(self.algorithm_comboBox.currentText())
        data = self.DataHolder.getInitialData()
        self.kmeans = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, 
                             max_iter=self.max_iter, tol=self.tol, algorithm=self.algorithm)
        
        km = self.kmeans.fit(data)
        self.setLabels(km.labels_)
        # print("calculator labels =" ,km.labels_)
        # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
        # cluster = np.hstack((ind,labels[:,None]))
        # cluster = cluster[cluster[:, -1].argsort()]
        # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
        self.setCenters(km.cluster_centers_)
        self.close()
        
    def setCenters(self,centers):
        self.kMeans_centers = centers
    def getCenters(self):
        return self.kMeans_centers    
    def setLabels(self,labels):
        self.kMeans_labels = labels
    def getLabels(self):
        return self.kMeans_labels
    
    def resetParameters(self):
        self.n_clusters_linedit.setText("8")
        self.init_comboBox.setCurrentIndex(0)
        self.n_init_lineEdit.setText("10")
        self.max_iter_lineEdit.setText("300")
        self.tol_lineEdit.setText("0.0001")
        self.algorithm_comboBox.setCurrentIndex(0)

class AffinityPropagationCalculator(AbstractCalculator,Ui_apWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
       
        self.setupUi(self)
        self.connectSignalSlots()
        self.DataHolder = DataHolder()
        self.ap_centers = []
        self.ap_labels = []
        
    def connectSignalSlots(self):
        self.OKButton.clicked.connect(self.calculate)
        self.resetButton.clicked.connect(self.resetParameters)

    def calculate(self):
        self.damping = float(self.damping_lineEdit.text())
        self.convergence_iter = int(self.convergence_iter_lineEdit.text())
        self.max_iter = int(self.max_iter_lineEdit.text())
        self.affinity = str(self.affinity_comboBox.currentText())
        data = self.DataHolder.getInitialData()
        self.ap = AffinityPropagation(damping=self.damping, convergence_iter=self.convergence_iter,
                                      max_iter=self.max_iter, affinity=self.affinity)
        ap = self.ap.fit(data)
        self.setLabels(ap.labels_)
        # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
        # cluster = np.hstack((ind,labels[:,None]))
        # cluster = cluster[cluster[:, -1].argsort()]
        # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
        self.setCenters(ap.cluster_centers_)
        self.close()
        
        
    def setCenters(self,centers):
        self.ap_centers = centers
    def getCenters(self):
        return self.ap_centers    
    def setLabels(self,labels):
        self.ap_labels = labels
    def getLabels(self):
        return self.ap_labels
    
    def resetParameters(self):
        self.damping_lineEdit.setText("0.5")
        self.convergence_iter_lineEdit.setText("15")
        self.max_iter_lineEdit.setText("200")
        self.affinity_comboBox.setCurrentIndex(0)
        
class meanShiftCalculator(AbstractCalculator,Ui_msWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
       
        self.setupUi(self)
        # self.show()
        # print(initialData)
        self.connectSignalSlots()
        self.DataHolder = DataHolder()
        self.ms_centers = []
        self.ms_labels = []
        
    def connectSignalSlots(self):
        self.OKButton.clicked.connect(self.calculate)
        self.resetButton.clicked.connect(self.resetParameters)

    def calculate(self):
        self.bandwidth = None if not self.bandwidth_lineEdit.text() else self.bandwidth_lineEdit.text()
        self.n_jobs = int(self.n_jobs_lineEdit.text())        
        self.max_iter = int(self.max_iter_lineEdit.text())
        data = self.DataHolder.getInitialData()
        self.ms = MeanShift(bandwidth=self.bandwidth, n_jobs=self.n_jobs, max_iter=self.max_iter)
        
        ms = self.ms.fit(data)
        self.setLabels(ms.labels_)
        # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
        # cluster = np.hstack((ind,labels[:,None]))
        # cluster = cluster[cluster[:, -1].argsort()]
        # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
        self.setCenters(ms.cluster_centers_)        
        
        self.close()
    def setCenters(self,centers):
        self.ms_centers = centers
    def getCenters(self):
        return self.ms_centers    
    def setLabels(self,labels):
        self.ms_labels = labels
    def getLabels(self):
        return self.ms_labels
    
    def resetParameters(self):        
        self.bandwidth_lineEdit.setText("")
        self.n_jobs_lineEdit.setText("1")        
        self.max_iter_lineEdit.setText("300")

class dbScanCalculator(AbstractCalculator,Ui_dbScanWindow):
     def __init__(self,parent=None):
         super().__init__(parent)
        
         self.setupUi(self)
         self.connectSignalSlots()
         self.DataHolder = DataHolder()
         self.dbs_centers = []
         self.dbs_labels = []
         
     def connectSignalSlots(self):
         self.OKButton.clicked.connect(self.calculate)
         self.resetButton.clicked.connect(self.resetParameters)

     def calculate(self):
         self.eps = float(self.eps_lineEdit.text())
         self.min_samples = int(self.min_samples_lineEdit.text())
         self.n_jobs = int(self.n_jobs_lineEdit.text())
         self.algorithm = str(self.algorithm_comboBox.currentText())
         data = self.DataHolder.getInitialData()
         self.dbs = DBSCAN(eps=self.eps, min_samples=self.min_samples,
                              n_jobs=self.n_jobs ,algorithm=self.algorithm)
         dbs = self.dbs.fit(data)
         self.setLabels(dbs.labels_)
         # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
         # cluster = np.hstack((ind,labels[:,None]))
         # cluster = cluster[cluster[:, -1].argsort()]
         # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
         # self.setCenters(dbs.cluster_centers_)
         
         self.close()
         
     def setCenters(self,centers=[]):
         self.dbs_centers = centers
     def getCenters(self):
         return self.dbs_centers    
     def setLabels(self,labels):
         self.dbs_labels = labels
     def getLabels(self):
         return self.dbs_labels
     
     def resetParameters(self):  
         self.eps_lineEdit.setText("0.5")
         self.min_samples_lineEdit.setText("5")
         self.n_jobs_lineEdit.setText("1")   
         self.algorithm_comboBox.setCurrentIndex(0)
        
class hcCalculator(AbstractCalculator,Ui_hcWindow):
     def __init__(self,parent=None):
         super().__init__(parent)
        
         self.setupUi(self)
         self.connectSignalSlots()
         self.DataHolder = DataHolder()
         self.hc_centers = []
         self.hc_labels = []
         
     def connectSignalSlots(self):
         self.OKButton.clicked.connect(self.calculate)
         self.resetButton.clicked.connect(self.resetParameters)

     def calculate(self):
         self.n_clusters = int(self.n_clusters_lineEdit.text())
         self.affinity = str(self.affinity_comboBox.currentText())
         self.linkage = str(self.linkage_comboBox.currentText())
         self.computeFullTree = str(self.computeFullTree_comboBox.currentText())
         data = self.DataHolder.getInitialData()
         self.hc = AgglomerativeClustering(n_clusters=self.n_clusters, affinity=self.affinity,
                                           linkage=self.linkage, compute_full_tree=self.computeFullTree)
         hc = self.hc.fit(data)
         self.setLabels(hc.labels_)
         # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
         # cluster = np.hstack((ind,labels[:,None]))
         # cluster = cluster[cluster[:, -1].argsort()]
         # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
         self.setCenters(hc.cluster_centers_)
         
         self.close()
         
     def setCenters(self,centers):
         self.hc_centers = centers
     def getCenters(self):
         return self.hc_centers    
     def setLabels(self,labels):
         self.hc_labels = labels
     def getLabels(self):
         return self.hc_labels
     
     def resetParameters(self):  
         self.n_clusters_lineEdit.setText("2")
         self.affinity_comboBox.setCurrentIndex(0)
         self.linkage_comboBox.setCurrentIndex(0)
         self.computeFullTree_comboBox.setCurrentIndex(0)
         
class scCalculator(AbstractCalculator,Ui_scWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
       
        self.setupUi(self)
        # self.show()
        # print(initialData)
        self.connectSignalSlots()
        self.DataHolder = DataHolder()
        self.sc_centers = []
        self.sc_labels = []
        
    def connectSignalSlots(self):
        self.OKButton.clicked.connect(self.calculate)
        self.resetButton.clicked.connect(self.resetParameters)

    def calculate(self):
        self.n_clusters = int(self.n_clusters_lineEdit.text())
        self.eigen_solver = str(self.eigen_solver_comboBox.currentText())
        self.n_init = int(self.n_init_lineEdit.text())
        self.n_components = int(self.n_components_lineEdit.text())
        self.gamma = float(self.gamma_lineEdit.text())
        self.assign_labels = str(self.assign_labels_comboBox.currentText())
        self.degree = int(self.degree_lineEdit.text())
        self.coef0 = float(self.coef0_lineEdit.text())
        self.n_jobs = int(self.n_jobs_lineEdit.text())
        data = self.DataHolder.getInitialData()
        self.sc = SpectralClustering(n_clusters=self.n_clusters, eigen_solver=self.eigen_solver,
                                     n_init=self.n_init, n_components=self.n_components, 
                                     gamma=self.gamma, assign_labels=self.assign_labels,
                                     degree=self.degree, coef0=self.coef0, n_jobs=self.n_jobs)
        
        sc = self.sc.fit(data)
        self.setLabels(sc.labels_)
        # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
        # cluster = np.hstack((ind,labels[:,None]))
        # cluster = cluster[cluster[:, -1].argsort()]
        # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
        # self.setCenters(sc.cluster_centers_)
        
        self.close()
        
    def setCenters(self,centers):
        self.sc_centers = centers
    def getCenters(self):
        return self.sc_centers    
    def setLabels(self,labels):
        self.sc_labels = labels
    def getLabels(self):
        return self.sc_labels
    
    def resetParameters(self):

        self.n_clusters_lineEdit.setText("8")
        self.eigen_solver_comboBox.setCurrentIndex(0)
        self.n_init_lineEdit.setText("10")
        self.n_components_lineEdit.setText("8")
        self.gamma_lineEdit.setText("1.0")
        self.assign_labels_comboBox.setCurrentIndex(0)
        self.degree_lineEdit.setText("3")
        self.coef0_lineEdit.setText("1")
        self.n_jobs_lineEdit.setText("1")
