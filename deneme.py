# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 02:22:20 2022

@author: piton
"""


from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import time
from DataHolder import DataHolder
from scipy.spatial import distance
from itertools import combinations
DH = DataHolder() 
def calculateClusters():
    # print("*********************calculate clusters******************")
    labels = DH.getLabels()
    n_clusters = len(np.unique(labels))
    DH.setNumberOfClusters(n_clusters)

    clusterList = []
    for i in range(0,(n_clusters)):
        clusterList.append(list(np.where(labels==i))[0])
    DH.setClusterNodes(clusterList)

def calculateCenters():    
    # print("*********************calculate centers******************")
    n_clusters = DH.getNumberOfClusters()
    data = DH.getInitialData()
    centers = []
    clusterList = DH.getClusterNodes()
    for i in range(0,n_clusters):
        x = np.mean(data[clusterList[i],0])
        y = np.mean(data[clusterList[i],1])
        centers.append([x,y])
    # print("centers = " ,centers)
      
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
    print("center nodes = " , center_nodes)   
    for i in range(0,n_clusters):  
        c = np.array(center_nodes[i])
        d = np.array(clusterList[i])    
        cluster_nodes.append(d[d != c])
    DH.setClusterNodes(cluster_nodes)
    DH.setCenterNodes(center_nodes)

def calculateFarhestDistance():

    center_nodes = DH.getCenterNodes()
    n_clusters = DH.getNumberOfClusters()
    farhest_dist = {}
    center_nodes_list = []
    dist_to_center_node = []
    clusterList = DH.getClusterNodes()
    for i in range(0,n_clusters):
        center_nodes_list.append(int(center_nodes[i]))
    for i in range(0,n_clusters):  
        # print("data center= ",data[center_nodes_list[i]])
        # print("data = ",data[clusterList[i]])
        a = data[center_nodes_list[i]]
        a = a.reshape(1,2)
        b = data[clusterList[i]]
        dist_to_center_node.append(distance.cdist( a, b, metric="euclidean" ))

    farhest_dist = dict.fromkeys(center_nodes_list,0)
    for i in range(0,n_clusters):
            farhest_dist[center_nodes_list[i]] = np.max(dist_to_center_node[i])            
    # print(farhest_dist)
    DH.setFarhestHubDistances(farhest_dist)
def calculatePairCombinations(): 

    center_nodes = DH.getCenterNodes()
    pair_combinations = combinations(list(center_nodes),2)
    pair_combinations = list(pair_combinations)

    DH.setPairCombinations(pair_combinations)
def calculatePairObjectives():

    pair_combinations = DH.getPairCombinations()
    data = DH.getInitialData()
    farhest_dist = DH.getFarhestHubDistances()

    pair_objectives = np.zeros(len(pair_combinations))
    for i in range(0,len(pair_combinations)):
        cluster_i = int(pair_combinations[i][0])
        cluster_j = int(pair_combinations[i][1])
        # print("farhest_dist = " ,farhest_dist)
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
    print("objective result =" ,objective_result)
    print()
    DH.setObjectiveResult(objective_result)
    # print(objective_result)
    
def RelocateHub():
   
    n_clusters = DH.getNumberOfClusters()
    cluster = DH.getClusterNodes()
    center_nodes = DH.getCenterNodes()
    cluster_nodes = DH.getClusterNodes()
    randIndex = int(np.random.randint(0,n_clusters,1))

    cluster = cluster_nodes[randIndex]
    nodeIndex = int(np.random.randint(0,len(cluster),1))
    hub = int(center_nodes[randIndex])
    center_nodes[randIndex] = np.array(cluster[nodeIndex])
    # print("center nodes after = ",center_nodes)
    cluster[nodeIndex] = np.array(hub)
    cluster_nodes[randIndex] = cluster
    print("cluster nodes = ",cluster_nodes)
    print("center nodes = ", center_nodes)

    DH.setClusterNodes(cluster_nodes)
    DH.setCenterNodes(center_nodes)

def SwapNodes():
   
    cluster = DH.getClusterNodes()   
    n_clusters = DH.getNumberOfClusters()
    print("clusters before = " ,cluster)
    randHub1 = int(np.random.randint(0,n_clusters,1))
    randHub2 = int(np.random.randint(0,n_clusters,1))
    while randHub1 == randHub2:
        randHub1 = int(np.random.randint(0,n_clusters,1))
    
    a = cluster[randHub1]
    b = cluster[randHub2]
    randIndex1 = int(np.random.randint(0,len(a),1))
    randIndex2 = int(np.random.randint(0,len(b),1))
    a[randIndex1],b[randIndex2] = b[randIndex2],a[randIndex1]
    print("clusters after = " ,cluster)

    
    DH.setClusterNodes(cluster)       

def ReallocateNode():   
    clusterList = DH.getClusterNodes() 
    n_clusters = DH.getNumberOfClusters()
    
    randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
    while randIndex1 == randIndex2:
        randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
    clusterList1 = list(clusterList[randIndex1])
    clusterList2 = list(clusterList[randIndex2])
    if not len(clusterList1) <=1:
        # print("hello")
        # print()
        print("cluster nodes before = ",clusterList)
        index = int(np.random.randint(0,len(clusterList1),1))
        temp = clusterList1.pop(int(index))
        clusterList2.append(temp)
        clusterList[randIndex1] = np.array(clusterList1)
        clusterList[randIndex2] = np.array(clusterList2)
        print("cluster nodes after = ",clusterList)
    DH.setClusterNodes(clusterList)       
    # print("end of reallocate")
    
    
    
df = pd.read_csv("C:/Users/piton/Desktop/projects/oop-proj2/data/10.txt",sep=" ",header=None)
df.columns = ["X","Y"]
data = np.array(df)


X = data
DH.setInitialData(X)
ac = KMeans(n_clusters=3).fit(X)
labels = ac.labels_
DH.setLabels(labels)
calculateClusters()
calculateCenters()
calculateCenterNodes()
calculateFarhestDistance()
calculatePairCombinations()
calculatePairObjectives()
 
limit = DH.getObjectiveResult()
i=0
resultss=[]
while limit >37000:
    select = int(np.random.randint(0,3,1))
    if select == 0:
        RelocateHub()
    if select == 1:
        ReallocateNode()
    if select == 2:
        SwapNodes()
        
    # indcl = DH.getClusterNodes()
    # limit = DH.getObjectiveResult()
    # centers = DH.getCenterNodes()
    # print(i," --->" ,limit,"---> ",centers,"-->" ,indcl)   
    calculateCenters()
    calculateFarhestDistance()
    calculatePairCombinations()
    calculatePairObjectives()  
    limit = DH.getObjectiveResult()

    
    plt.pause(0.05)
    i+=1
    print(" iterasyon number = ", i)

plt.show()    

def printGraph():  
    data = DH.getInitialData()
    centers = DH.getCenters()
    labels = DH.getLabels()
    initialSolution_figure = plt.figure()
    initialSolution_canvas = FigureCanvas(initialSolution_figure)
            
    # self.initialSolution_figure.clear()
    ploting = initialSolution_figure.add_subplot(111)
    
    if len(data):
        ploting.scatter(data[:,0], data[:,1],color="k",s=20) 
        # for i in range(len(self.__data)):
        #     plt.annotate(str(i),(self.__data[:,0], self.__data[:,1]))
        # print("data")
    
    if len(labels):
        ploting.scatter(data[:,0], data[:,1],c = labels,s = 20,cmap = 'rainbow')
        # for i in range(len(self.__data)):
        #     plt.annotate(str(i),(self.__data[:,0], self.__data[:,1]))
        # print("lbl")
    if len(centers):
        ploting.scatter(np.array(centers)[:, 0],np.array(centers)[:, 1],c = "red",s = 100, marker="x",alpha = 1,linewidth=1)
        # print("center")


# centers = DH.getCenters()
# labels = DH.getLabels()
# printGraph()    
    
    
    