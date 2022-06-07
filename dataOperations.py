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

# from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from scipy.spatial import distance
from itertools import combinations

import numpy as np


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
        labels = km.labels_
        self.DataHolder.setLabels(km.labels_)
        calculateClusters()
        calculateCenters()
        calculateCenterNodes()
        calculateFarhestDistance()
        calculatePairCombinations()
        calculatePairObjectives()
        # self.setLabels(km.labels_)
        
        # cluster = [0]*self.n_clusters
        # for i in range(0,(self.n_clusters)):
        #     cluster[i] = (np.where(km.labels_==i))
        # self.DataHolder.setClusterIndices(cluster)    
        
        # self.setCenters(km.cluster_centers_)
        
        # self.DataHolder.setCenters(km.cluster_centers_)
        
        # center_nodes = []  
        # centers = km.cluster_centers_ 
        # dist = distance.cdist( centers, data, metric="euclidean" )
        # for i in range(0,self.n_clusters):
        #     center_nodes.append(np.where(dist[i] == min(dist[i])))

        # center_nodes = np.array(center_nodes)
        # center_nodes = center_nodes.reshape(1,self.n_clusters)
        # center_nodes = list(*center_nodes)    
        # self.DataHolder.setCenterNodes(center_nodes)
        
        # farhest_dist = {}
        # farhest_dist = dict.fromkeys(center_nodes, 0)
        # for i in range(0,self.n_clusters):
        #     farhest_dist[center_nodes[i]] = (max(dist[i]))            
        # self.DataHolder.setFarhestHubDistances(farhest_dist)
        
        # pair_combinations = combinations(list(center_nodes),2)
        # pair_combinations = list(pair_combinations)
        # self.DataHolder.setPairCombinations(pair_combinations)
        
        # pair_objectives = np.zeros(len(pair_combinations))
        # for i in range(0,len(pair_combinations)):
        #     cluster_i = pair_combinations[i][0]
        #     cluster_j = pair_combinations[i][1]
        #     dihi = farhest_dist[cluster_i]
            
        #     p1 = data[cluster_i]
        #     p1 = p1.reshape(1,2)
        #     p2 = data[cluster_j]
        #     p2 = p2.reshape(1,2)
        #     dhihj = distance.cdist(p1 ,p2,metric="euclidean")
        #     djhj = farhest_dist[cluster_j]
        #     objij = dihi+0.75*dhihj+djhj
        #     pair_objectives[i] = (objij)
        
        # self.DataHolder.setPairObjectives(pair_objectives)
        # objective_result = max(pair_objectives)
        # self.DataHolder.setObjectiveResult(objective_result)
        
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
        labels = ap.labels_
        self.DataHolder.setLabels(ap.labels_)
        calculateClusters()
        calculateCenters()
        calculateCenterNodes()
        calculateFarhestDistance()
        calculatePairCombinations()
        calculatePairObjectives()
        # self.setLabels(ap.labels_)
        
        # self.n_clusters = len(np.unique(ap.labels_))
        # cluster = [0]*self.n_clusters
        # for i in range(0,(self.n_clusters)):
        #     cluster[i] = (np.where(ap.labels_==i))
        # self.DataHolder.setClusterIndices(cluster)
        # # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
        # # cluster = np.hstack((ind,labels[:,None]))
        # # cluster = cluster[cluster[:, -1].argsort()]
        # # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
        # self.setCenters(ap.cluster_centers_)
        # self.DataHolder.setCenters(ap.cluster_centers_)
        
        # center_nodes = []  
        # centers = ap.cluster_centers_ 
        # dist = distance.cdist( centers, data, metric="euclidean" )
        # for i in range(0,self.n_clusters):
        #     center_nodes.append(np.where(dist[i] == min(dist[i])))

        # center_nodes = np.array(center_nodes)
        # center_nodes = center_nodes.reshape(1,self.n_clusters)
        # center_nodes = list(*center_nodes)    
        # self.DataHolder.setCenterNodes(center_nodes)
        
        # farhest_dist = {}
        # farhest_dist = dict.fromkeys(center_nodes, 0)
        # for i in range(0,self.n_clusters):
        #     farhest_dist[center_nodes[i]] = (max(dist[i]))            
        # self.DataHolder.setFarhestHubDistances(farhest_dist)
        
        # pair_combinations = combinations(list(center_nodes),2)
        # pair_combinations = list(pair_combinations)
        # self.DataHolder.setPairCombinations(pair_combinations)
        
        # pair_objectives = np.zeros(len(pair_combinations))
        # for i in range(0,len(pair_combinations)):
        #     cluster_i = pair_combinations[i][0]
        #     cluster_j = pair_combinations[i][1]
        #     dihi = farhest_dist[cluster_i]
            
        #     p1 = data[cluster_i]
        #     p1 = p1.reshape(1,2)
        #     p2 = data[cluster_j]
        #     p2 = p2.reshape(1,2)
        #     dhihj = distance.cdist(p1 ,p2,metric="euclidean")
        #     djhj = farhest_dist[cluster_j]
        #     objij = dihi+0.75*dhihj+djhj
        #     pair_objectives[i] = (objij)
        
        # self.DataHolder.setPairObjectives(pair_objectives)
        # objective_result = max(pair_objectives)
        # self.DataHolder.setObjectiveResult(objective_result)
        
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
        labels = ms.labels_
        self.DataHolder.setLabels(ms.labels_)
        calculateClusters()
        calculateCenters()
        calculateCenterNodes()
        calculateFarhestDistance()
        calculatePairCombinations()
        calculatePairObjectives()
        
        # self.setLabels(ms.labels_)
        
        # self.n_clusters = len(np.unique(ms.labels_))
        # cluster = [0]*self.n_clusters
        # for i in range(0,(self.n_clusters)):
        #     cluster[i] = (np.where(ms.labels_==i))
        # self.DataHolder.setClusterIndices(cluster)
        # # ind = np.reshape(np.arange(0,len(labels)),[len(labels),1])
        # # cluster = np.hstack((ind,labels[:,None]))
        # # cluster = cluster[cluster[:, -1].argsort()]
        # # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
        # self.setCenters(ms.cluster_centers_)  
        # self.DataHolder.setCenters(ms.cluster_centers_)
        # center_nodes = []  
        # centers = ms.cluster_centers_ 
        # dist = distance.cdist( centers, data, metric="euclidean" )
        # for i in range(0,self.n_clusters):
        #     center_nodes.append(np.where(dist[i] == min(dist[i])))

        # center_nodes = np.array(center_nodes)
        # center_nodes = center_nodes.reshape(1,self.n_clusters)
        # center_nodes = list(*center_nodes)    
        # self.DataHolder.setCenterNodes(center_nodes)
        
        # farhest_dist = {}
        # farhest_dist = dict.fromkeys(center_nodes, 0)
        # for i in range(0,self.n_clusters):
        #     farhest_dist[center_nodes[i]] = (max(dist[i]))            
        # self.DataHolder.setFarhestHubDistances(farhest_dist)
        
        # pair_combinations = combinations(list(center_nodes),2)
        # pair_combinations = list(pair_combinations)
        # self.DataHolder.setPairCombinations(pair_combinations)
        
        # pair_objectives = np.zeros(len(pair_combinations))
        # for i in range(0,len(pair_combinations)):
        #     cluster_i = pair_combinations[i][0]
        #     cluster_j = pair_combinations[i][1]
        #     dihi = farhest_dist[cluster_i]
            
        #     p1 = data[cluster_i]
        #     p1 = p1.reshape(1,2)
        #     p2 = data[cluster_j]
        #     p2 = p2.reshape(1,2)
        #     dhihj = distance.cdist(p1 ,p2,metric="euclidean")
        #     djhj = farhest_dist[cluster_j]
        #     objij = dihi+0.75*dhihj+djhj
        #     pair_objectives[i] = (objij)
        
        # self.DataHolder.setPairObjectives(pair_objectives)
        # objective_result = max(pair_objectives)
        # self.DataHolder.setObjectiveResult(objective_result)
        
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
         # ---------------------------------
         # centers = [[1, 1], [-1, -1], [1, -1]]
         # data, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)   
                            
         # data = StandardScaler().fit_transform(data)
         # ---------------------------------
         self.eps = float(self.eps_lineEdit.text())
         self.min_samples = int(self.min_samples_lineEdit.text())
         self.n_jobs = int(self.n_jobs_lineEdit.text())
         self.algorithm = str(self.algorithm_comboBox.currentText())
         data = self.DataHolder.getInitialData()
         self.dbs = DBSCAN(eps=self.eps, min_samples=self.min_samples,
                              n_jobs=self.n_jobs ,algorithm=self.algorithm)
         dbs = self.dbs.fit(data)
         labels = dbs.labels_
         self.DataHolder.setLabels(dbs.labels_)
         calculateClusters()
         calculateCenters()
         calculateCenterNodes()
         calculateFarhestDistance()
         calculatePairCombinations()
         calculatePairObjectives()
         # core_samples_mask = np.zeros_like(dbs.labels_, dtype=bool)
         # core_samples_mask[dbs.core_sample_indices_] = True
         
         
         # cluster = [0]*self.n_clusters
         # for i in range(0,(self.n_clusters)):
         #     cluster[i] = (np.where(dbs.labels_==i))
         # self.DataHolder.setClusterIndices(cluster)
         # print("lbl: ",dbs.labels_)
         # self.n_clusters_ = len(set(dbs.labels_)) - (1 if -1 in dbs.labels_ else 0)
         # self.n_noise_ = list(dbs.labels_).count(-1)
         # print()
         # print(self.n_noise_)
         # print()
         # print(self.n_clusters_)
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
         # self.setLabels(hc.labels_)
         
         labels = hc.labels_
         self.DataHolder.setLabels(hc.labels_)
         calculateClusters()
         calculateCenters()
         calculateCenterNodes()
         calculateFarhestDistance()
         calculatePairCombinations()
         calculatePairObjectives()
         
         # n_clusters = hc.n_clusters_
         # print("labels:")
         # print(labels)
         # print()
         # print("number of clusters: ",n_clusters)
            
         # cluster = [0]*n_clusters
         # for i in range(0,(n_clusters)):
         #     cluster[i] = (np.where(labels==i))
         # for i in range(0,(n_clusters)):
         #     print("cluster ",i,"-->",*cluster[i])
            
         # centers = [0]*n_clusters
         # for i in range(0,n_clusters):
         #     x = np.mean(data[np.where(labels==i)][:,0])
         #     y = np.mean(data[np.where(labels==i)][:,1])
         #     centers[i] = [x,y]
        
         # print("centers: ")
         # print(centers)
        
         # center_nodes = []  
         # dist = distance.cdist(centers, data, metric="euclidean" )
         # for i in range(0,self.n_clusters):
         #    center_nodes.append(np.where(dist[i] == min(dist[i])))
        
         # center_nodes = np.array(center_nodes)
         # center_nodes = center_nodes.reshape(1,n_clusters)
         # center_nodes = list(*center_nodes)    
         # print("Cluster center nodes -->",*center_nodes)
        
         # farhest_dist = {}
         # farhest_dist = dict.fromkeys(center_nodes, 0)
         # for i in range(0,n_clusters):
         #     farhest_dist[center_nodes[i]] = (max(dist[i]))
        
         # print("****Farhest hub distances****")
         # print(farhest_dist)
        
         # pair_combinations = combinations(list(center_nodes),2)
         # pair_combinations = list(pair_combinations)
         # print("possible pair combinations --> ",pair_combinations)
        
        
         # pair_objectives = np.zeros(len(pair_combinations))
         # for i in range(0,len(pair_combinations)):
         #     cluster_i = pair_combinations[i][0]
         #     cluster_j = pair_combinations[i][1]
         #     dihi = farhest_dist[cluster_i]
            
         #     p1 = data[cluster_i]
         #     p1 = p1.reshape(1,2)
         #     p2 = data[cluster_j]
         #     p2 = p2.reshape(1,2)
         #     dhihj = distance.cdist(p1 ,p2,metric="euclidean")
         #     djhj = farhest_dist[cluster_j]
         #     objij = dihi+0.75*dhihj+djhj
         #     pair_objectives[i] = (objij)
            
         #     objective_result = max(pair_objectives)
        
         # print("****Pair objectives****")
         # print(pair_objectives)
         # print()
         # print("Objective result --> " ,objective_result)
         # cluster = np.hstack((ind,labels[:,None]))
         # cluster = cluster[cluster[:, -1].argsort()]
         # cluster = np.split(cluster[:,:-1], np.unique(cluster[:, -1], return_index=True)[1][1:])
         # self.setCenters(hc.cluster_centers_)
         
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
        labels = sc.labels_
        self.DataHolder.setLabels(sc.labels_)
        calculateClusters()
        calculateCenters()
        calculateCenterNodes()
        calculateFarhestDistance()
        calculatePairCombinations()
        calculatePairObjectives()
        
        # self.setLabels(sc.labels_)
        
        # labels
        # cluster = [0]*self.n_clusters
        # for i in range(0,(self.n_clusters)):
        #     cluster[i] = (np.where(sc.labels_==i))
        # self.DataHolder.setClusterIndices(cluster)
        
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

def calculateClusters():
    DH = DataHolder()
    labels = DH.getLabels()
    n_clusters = len(np.unique(labels))
    DH.setNumberOfClusters(n_clusters)
    cluster = [0]*n_clusters
    for i in range(0,(n_clusters)):
        cluster[i] = (np.where(labels==i))
    DH.setClusterIndices(cluster)
def calculateCenters():    
    DH = DataHolder()
    n_clusters = DH.getNumberOfClusters()
    data = DH.getInitialData()
    labels = DH.getLabels()
    centers = [0]*n_clusters
    for i in range(0,n_clusters):
        x = np.mean(data[np.where(labels==i)][:,0])
        y = np.mean(data[np.where(labels==i)][:,1])
        centers[i] = [x,y]
    DH.setCenters(centers)
def calculateCenterNodes():
    DH = DataHolder()
    center_nodes = []   
    data = DH.getInitialData()
    centers = DH.getCenters()
    n_clusters = DH.getNumberOfClusters()
        
    dist = distance.cdist( centers, data, metric="euclidean" )
    DH.setDistanceMatrix(dist)
    for i in range(0,n_clusters):
        center_nodes.append(np.where(dist[i] == min(dist[i])))

    center_nodes = np.array(center_nodes)
    center_nodes = center_nodes.reshape(1,n_clusters)
    center_nodes = list(*center_nodes)    
    DH.setCenterNodes(center_nodes)

def calculateFarhestDistance():
    DH = DataHolder()
    center_nodes = DH.getCenterNodes()
    n_clusters = DH.getNumberOfClusters()
    dist = DH.getDistanceMatrix()
    farhest_dist = {}
    farhest_dist = dict.fromkeys(center_nodes, 0)
    for i in range(0,n_clusters):
        farhest_dist[center_nodes[i]] = (max(dist[i]))            
    DH.setFarhestHubDistances(farhest_dist)
def calculatePairCombinations(): 
    DH = DataHolder()
    center_nodes = DH.getCenterNodes()
    pair_combinations = combinations(list(center_nodes),2)
    pair_combinations = list(pair_combinations)
    DH.setPairCombinations(pair_combinations)
def calculatePairObjectives():
    DH = DataHolder()
    pair_combinations = DH.getPairCombinations()
    data = DH.getInitialData()
    farhest_dist = DH.getFarhestHubDistances()
    pair_objectives = np.zeros(len(pair_combinations))
    for i in range(0,len(pair_combinations)):
        cluster_i = pair_combinations[i][0]
        cluster_j = pair_combinations[i][1]
        dihi = farhest_dist[cluster_i]
        
        p1 = data[cluster_i]
        p1 = p1.reshape(1,2)
        p2 = data[cluster_j]
        p2 = p2.reshape(1,2)
        dhihj = distance.cdist(p1 ,p2,metric="euclidean")
        djhj = farhest_dist[cluster_j]
        objij = dihi+0.75*dhihj+djhj
        pair_objectives[i] = (objij)
    
    DH.setPairObjectives(pair_objectives)
    objective_result = max(pair_objectives)
    DH.setObjectiveResult(objective_result)