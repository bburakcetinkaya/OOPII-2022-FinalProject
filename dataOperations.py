# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:19:55 2022

@author: Burak Çetinkaya
        151220152110
"""
from DataHolder import DataHolder
from PyQt5 import QtWidgets,QtCore
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


from scipy.spatial import distance
from itertools import combinations

import numpy as np
import time

from abc import ABCMeta, abstractmethod
DH = DataHolder()
def calculateClusters():
    
    labels = DH.getLabels()
    n_clusters = len(np.unique(labels))
    DH.setNumberOfClusters(n_clusters)
    
    clusterList = []
    for i in range(0,(n_clusters)):
        clusterList.append(list(np.where(labels==i))[0])
    #print("clusterList ", clusterList)
    DH.setClusterNodes(clusterList)
def calculateCenters():    
 
    n_clusters = DH.getNumberOfClusters()
    data = DH.getInitialData()
    centers = []
    clusterList = DH.getClusterNodes()
    for i in range(0,n_clusters):
        x = np.mean(data[clusterList[i],0])
        y = np.mean(data[clusterList[i],1])
        centers.append([x,y])
    #print("centers: ", centers)  
    DH.setCenters(centers)
def calculateCenterNodes():

    center_nodes = []  
    cluster_nodes = []
    data = DH.getInitialData()
    centers = DH.getCenters()
    n_clusters = DH.getNumberOfClusters()
    clusterList = DH.getClusterNodes()
        
    dist = distance.cdist(centers, data, metric="euclidean" )
    DH.setDistanceMatrix(dist)
    for i in range(0,n_clusters):
        center_nodes.append(np.where(dist[i] == min(dist[i]))[0])
    # #print("center nodes = " , center_nodes)   
    for i in range(0,n_clusters):  
        c = np.array(center_nodes[i])
        d = np.array(clusterList[i])    
        cluster_nodes.append(d[d != c])
    #print("cluster_nodes: ", cluster_nodes)  
    #print("center_nodes: ", center_nodes)  
    DH.setClusterNodes(cluster_nodes)
    DH.setCenterNodes(center_nodes)

def calculateFarhestDistance():

    data = DH.getInitialData()
    center_nodes = DH.getCenterNodes()
    n_clusters = DH.getNumberOfClusters()
    farhest_dist = {}
    center_nodes_list = []
    dist_to_center_node = []
    clusterList = DH.getClusterNodes()
    for i in range(0,n_clusters):
        center_nodes_list.append(int(center_nodes[i]))
    for i in range(0,n_clusters):  
        a = data[center_nodes_list[i]]
        a = a.reshape(1,2)
        b = data[clusterList[i]]
        dist_to_center_node.append(distance.cdist( a, b, metric="euclidean" ))

    farhest_dist = dict.fromkeys(center_nodes_list,0)
    for i in range(0,n_clusters):
        farhest_dist[center_nodes_list[i]] = np.max(dist_to_center_node[i])            
    #print("farhest_dist: ", farhest_dist)
    DH.setFarhestHubDistances(farhest_dist)
def calculatePairCombinations(): 

    center_nodes = DH.getCenterNodes()
    pair_combinations = combinations(list(center_nodes),2)
    pair_combinations = list(pair_combinations)
    #print("pair_combinations: ", pair_combinations)
    DH.setPairCombinations(pair_combinations)
def calculatePairObjectives():

    pair_combinations = DH.getPairCombinations()
    data = DH.getInitialData()
    farhest_dist = DH.getFarhestHubDistances()

    pair_objectives = np.zeros(len(pair_combinations))
    for i in range(0,len(pair_combinations)):
        cluster_i = int(pair_combinations[i][0])
        cluster_j = int(pair_combinations[i][1])
        # #print("farhest_dist = " ,farhest_dist)
        dihi = farhest_dist[cluster_i]        
        p1 = data[cluster_i]
        p1 = p1.reshape(1,2)
        p2 = data[cluster_j]
        p2 = p2.reshape(1,2)
        dhihj = distance.cdist(p1 ,p2,metric="euclidean")
        djhj = farhest_dist[cluster_j]
        objij = dihi+2*dhihj+djhj
        pair_objectives[i] = (objij)
    #print("pair_objectives: ", pair_objectives)
    DH.setPairObjectives(pair_objectives)
    objective_result = max(pair_objectives)
    #print("objective_result: ", objective_result)
    DH.setObjectiveResult(objective_result)
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
class SimulatedAnneling(object):
    def __init__(self,n_iterations):
        self.__n_iterations = n_iterations
        self.DH = DataHolder()
        self.calculate()
    def calculate(self): 
        self.progress = QtWidgets.QProgressDialog()
        self.progress.setMinimum(0)
        self.progress.setMaximum(self.__n_iterations)
        self.progress.setLabelText("Processing")
        self.progress.resize(400,70)
        self.progress.findChildren(QtWidgets.QPushButton)[0].hide()
        self.progress.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
        self.progress.setWindowModality(QtCore.Qt.ApplicationModal)
        self.progress.setFixedSize(self.progress.geometry().width(),self.progress.geometry().height())
        self.progress.show();
        best_solution = self.DH.getObjectiveResult()  
        start = time.time()
        for i in range (0,self.__n_iterations):
            self.progress.setValue(i);
            calculateCenters()
            calculateFarhestDistance()
            calculatePairCombinations()
            calculatePairObjectives()
            solution = self.DH.getObjectiveResult()
            if solution < 0.97*best_solution:
                best_solution = solution 
                self.DH.setResultIterationsNumber(i)
                # print(555)
                self.setBestSolutionResults()
            self.operate()
        self.DH.setTotalIterations(self.__n_iterations)
        stop = time.time()
        duration = stop-start
        self.DH.setExecutionTime(duration)
        
        self.progress.close()
        self.progress.deleteLater()
    
    def operate(self):
        select = int(np.random.randint(0,3,1))
        if select == 0:
            print(0)
            self.RelocateHub()
        if select == 1:
            print(1)
            self.ReallocateNode()
        if select == 2:
            print(2)
            self.SwapNodes()
            
    def RelocateHub(self):       
        n_clusters = self.DH.getNumberOfClusters()
        cluster = self.DH.getClusterNodes()
        ##print("cluster before: ", cluster)
        center_nodes = self.DH.getCenterNodes()
        cluster_nodes = self.DH.getClusterNodes()
        randIndex = int(np.random.randint(0,n_clusters,1))

        cluster = cluster_nodes[randIndex]
        nodeIndex = int(np.random.randint(0,len(cluster),1))
        hub = int(center_nodes[randIndex])
        center_nodes[randIndex] = np.array(cluster[nodeIndex])
        cluster[nodeIndex] = np.array(hub)
        cluster_nodes[randIndex] = cluster
        self.DH.setClusterNodes(cluster_nodes)
        self.DH.setCenterNodes(center_nodes)
        ##print("cluster after: ", cluster_nodes)
    def SwapNodes(self):       
        cluster = self.DH.getClusterNodes()  
        ##print("before swap:",self.DH.getClusterNodes())
        n_clusters = self.DH.getNumberOfClusters()
        randHub1 = int(np.random.randint(0,n_clusters,1))
        randHub2 = int(np.random.randint(0,n_clusters,1))
        while randHub1 == randHub2:
            randHub1 = int(np.random.randint(0,n_clusters,1))
        
        a = cluster[randHub1]
        b = cluster[randHub2]
        randIndex1 = int(np.random.randint(0,len(a),1))
        randIndex2 = int(np.random.randint(0,len(b),1))
        a[randIndex1],b[randIndex2] = b[randIndex2],a[randIndex1]
        self.DH.setClusterNodes(cluster)  
        ##print("after swap:",self.DH.getClusterNodes())

    def ReallocateNode(self):   
        clusterList = self.DH.getClusterNodes() 
        n_clusters = self.DH.getNumberOfClusters()
        
        randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
        while randIndex1 == randIndex2:
            randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
        clusterList1 = list(clusterList[randIndex1])
        clusterList2 = list(clusterList[randIndex2])
        if not len(clusterList1) <=1:
            index = int(np.random.randint(0,len(clusterList1),1))
            temp = clusterList1.pop(int(index))
            clusterList2.append(temp)
            clusterList[randIndex1] = np.array(clusterList1)
            clusterList[randIndex2] = np.array(clusterList2)
        self.DH.setClusterNodes(clusterList)  
        
    def setBestSolutionResults(self):
        self.DH.setBestCenterNodes(self.DH.getCenterNodes())        
        self.DH.setBestCenters(self.DH.getCenters())
        self.DH.setBestClusterNodes(self.DH.getClusterNodes())
        self.DH.setBestDistanceMatrix(self.DH.getDistanceMatrix())
        self.DH.setBestFarhestHubDistances(self.DH.getFarhestHubDistances())
        self.DH.setBestPairCombinations(self.DH.getPairCombinations())
        self.DH.setBestPairObjectives(self.DH.getPairObjectives())
        self.DH.setBestObjectiveResult(self.DH.getObjectiveResult())

class HillClimbing(object):
    def __init__(self,n_iterations):
        self.__n_iterations = n_iterations
        self.DH = DataHolder()
        self.calculate()
    def calculate(self):   
        self.progress = QtWidgets.QProgressDialog()
        self.progress.setMinimum(0)
        self.progress.setMaximum(self.__n_iterations)
        self.progress.setLabelText("Processing")
        self.progress.resize(400,70)
        self.progress.findChildren(QtWidgets.QPushButton)[0].hide()
        self.progress.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
        self.progress.setWindowModality(QtCore.Qt.ApplicationModal)
        self.progress.setFixedSize(self.progress.geometry().width(),self.progress.geometry().height())
        self.progress.show();
        best_solution = self.DH.getObjectiveResult()   
        start = time.time()
        for i in range (0,self.__n_iterations):
            self.progress.setValue(i);
            calculateCenters()
            calculateFarhestDistance()
            calculatePairCombinations()
            calculatePairObjectives()
            solution = self.DH.getObjectiveResult()
            if solution < best_solution:
                best_solution = solution  
                self.DH.setResultIterationsNumber(i)
                self.setBestSolutionResults()
            self.operate()
        self.DH.setTotalIterations(self.__n_iterations)
        stop = time.time()
        duration = stop-start
        self.DH.setExecutionTime(duration)
        self.progress.close()
        self.progress.deleteLater()

    def operate(self):
        select = int(np.random.randint(0,3,1))
        if select == 0:
            print(0)
            self.RelocateHub()
        if select == 1:
            print(1)
            self.ReallocateNode()
        if select == 2:
            print(2)
            self.SwapNodes()
            
    def RelocateHub(self):       
        n_clusters = self.DH.getNumberOfClusters()
        cluster = self.DH.getClusterNodes()
        ##print("cluster before: ", cluster)
        center_nodes = self.DH.getCenterNodes()
        cluster_nodes = self.DH.getClusterNodes()
        randIndex = int(np.random.randint(0,n_clusters,1))

        cluster = cluster_nodes[randIndex]
        nodeIndex = int(np.random.randint(0,len(cluster),1))
        hub = int(center_nodes[randIndex])
        center_nodes[randIndex] = np.array(cluster[nodeIndex])
        cluster[nodeIndex] = np.array(hub)
        cluster_nodes[randIndex] = cluster
        self.DH.setClusterNodes(cluster_nodes)
        self.DH.setCenterNodes(center_nodes)
        ##print("cluster after: ", cluster_nodes)
    def SwapNodes(self):       
        cluster = self.DH.getClusterNodes()  
        ##print("before swap:",self.DH.getClusterNodes())
        n_clusters = self.DH.getNumberOfClusters()
        randHub1 = int(np.random.randint(0,n_clusters,1))
        randHub2 = int(np.random.randint(0,n_clusters,1))
        while randHub1 == randHub2:
            randHub1 = int(np.random.randint(0,n_clusters,1))
        
        a = cluster[randHub1]
        b = cluster[randHub2]
        randIndex1 = int(np.random.randint(0,len(a),1))
        randIndex2 = int(np.random.randint(0,len(b),1))
        a[randIndex1],b[randIndex2] = b[randIndex2],a[randIndex1]
        self.DH.setClusterNodes(cluster)  
        ##print("after swap:",self.DH.getClusterNodes())

    def ReallocateNode(self):   
        clusterList = self.DH.getClusterNodes() 
        n_clusters = self.DH.getNumberOfClusters()
        
        randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
        while randIndex1 == randIndex2:
            randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
        clusterList1 = list(clusterList[randIndex1])
        clusterList2 = list(clusterList[randIndex2])
        if not len(clusterList1) <=1:
            index = int(np.random.randint(0,len(clusterList1),1))
            temp = clusterList1.pop(int(index))
            clusterList2.append(temp)
            clusterList[randIndex1] = np.array(clusterList1)
            clusterList[randIndex2] = np.array(clusterList2)
        self.DH.setClusterNodes(clusterList)  
        
    def setBestSolutionResults(self):
        self.DH.setBestCenterNodes(self.DH.getCenterNodes())        
        self.DH.setBestCenters(self.DH.getCenters())
        self.DH.setBestClusterNodes(self.DH.getClusterNodes())
        self.DH.setBestDistanceMatrix(self.DH.getDistanceMatrix())
        self.DH.setBestFarhestHubDistances(self.DH.getFarhestHubDistances())
        self.DH.setBestPairCombinations(self.DH.getPairCombinations())
        self.DH.setBestPairObjectives(self.DH.getPairObjectives())
        self.DH.setBestObjectiveResult(self.DH.getObjectiveResult())
        print(999999999999999999999999999999999999)
        
class KMeansCalculator(AbstractCalculator,Ui_kMeansWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
       
        self.setupUi(self)
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
        self.DataHolder.setLabels(labels)
        # calculateParameters() 
        calculateClusters()
        calculateCenters()
        calculateCenterNodes()
        calculateFarhestDistance()
        calculatePairCombinations()
        calculatePairObjectives()
        self.close()

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
        self.DataHolder.setLabels(labels)
        calculateClusters()
        calculateCenters()
        calculateCenterNodes()
        calculateFarhestDistance()
        calculatePairCombinations()
        calculatePairObjectives()
        self.close()     
        
    def resetParameters(self):
        self.damping_lineEdit.setText("0.5")
        self.convergence_iter_lineEdit.setText("15")
        self.max_iter_lineEdit.setText("200")
        self.affinity_comboBox.setCurrentIndex(0)
        
class meanShiftCalculator(AbstractCalculator,Ui_msWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
       
        self.setupUi(self)
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
        self.DataHolder.setLabels(labels)
        calculateClusters()
        calculateCenters()
        calculateCenterNodes()
        calculateFarhestDistance()
        calculatePairCombinations()
        calculatePairObjectives()
        self.close()

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
         labels = dbs.labels_
         self.DataHolder.setLabels(labels)
         calculateClusters()
         calculateCenters()
         calculateCenterNodes()
         calculateFarhestDistance()
         calculatePairCombinations()
         calculatePairObjectives()

         self.close()

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
         self.DataHolder.setLabels(labels)
         calculateClusters()
         calculateCenters()
         calculateCenterNodes()
         calculateFarhestDistance()
         calculatePairCombinations()
         calculatePairObjectives()           
         self.close()

     def resetParameters(self):  
         self.n_clusters_lineEdit.setText("2")
         self.affinity_comboBox.setCurrentIndex(0)
         self.linkage_comboBox.setCurrentIndex(0)
         self.computeFullTree_comboBox.setCurrentIndex(0)
         
class scCalculator(AbstractCalculator,Ui_scWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
       
        self.setupUi(self)
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
        self.DataHolder.setLabels(labels)
        calculateClusters()
        calculateCenters()
        calculateCenterNodes()
        calculateFarhestDistance()
        calculatePairCombinations()
        calculatePairObjectives()
        self.close()
        
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

# class calculateParameters(object):
# def __init__(self):
#     self.DH = DataHolder()
#     self.calculateClusters()
#     self.calculateCenters()
#     self.calculateCenterNodes()
#     self.calculateFarhestDistance()
#     self.calculatePairCombinations()
#     self.calculatePairObjectives()

