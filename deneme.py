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
    print("*********************calculate clusters******************")
    labels = DH.getLabels()
    n_clusters = len(np.unique(labels))
    DH.setNumberOfClusters(n_clusters)

    clusterList = []
    for i in range(0,(n_clusters)):
        clusterList.append(list(np.where(labels==i))[0])
    DH.setClusterIndices(clusterList)

def calculateCenters():    
    print("*********************calculate centers******************")
    n_clusters = DH.getNumberOfClusters()
    data = DH.getInitialData()
    centers = []
    clusterList = DH.getClusterIndices()
    for i in range(0,n_clusters):
        x = np.mean(data[[clusterList[i]],0])
        y = np.mean(data[[clusterList[i]],1])
        centers.append([x,y])
    print("centers = " ,centers)
      
    DH.setCenters(centers)
def calculateCenterNodes():

    center_nodes = []   
    data = DH.getInitialData()
    centers = DH.getCenters()
    n_clusters = DH.getNumberOfClusters()
        
    dist = distance.cdist(centers, data, metric="euclidean" )
    DH.setDistanceMatrix(dist)
    for i in range(0,n_clusters):
        center_nodes.append(np.where(dist[i] == min(dist[i]))[0])
    print("center nodes = " , center_nodes)      
    DH.setCenterNodes(center_nodes)

def calculateFarhestDistance():

    center_nodes = DH.getCenterNodes()
    n_clusters = DH.getNumberOfClusters()
    farhest_dist = {}
    center_nodes_list = []
    dist_to_center_node = []
    clusterList = DH.getClusterIndices()
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
            farhest_dist[center_nodes_list[i]] = (max(dist_to_center_node[i][0]))            
    print(farhest_dist)
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
        print("farhest_dist = " ,farhest_dist)
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
    
# def RelocateHub():
#     DH = DataHolder()
#     n_clusters = DH.getNumberOfClusters()  
#     center_nodes = DH.getCenterNodes()
#     cluster = DH.getClusterIndices()
#     for i in range(0,n_clusters):
#         # listeden center node lar silinecek öyle çalışır bence :) kolay gelsin koçum benim beeeee 
#         hub = center_nodes[i]
#         print("hub -> ",hub)
#         center_nodes[i] = np.random.choice(*cluster[i])
#         # index = np.where
#         print("center nodes at ",i," -> ",center_nodes[i])
#         index_of_cluster = np.where(cluster[i] == center_nodes[i])
#         cluster[i][0][index_of_cluster[1]] = hub
#         print("indx opfsdfgn ")
#         print(index_of_cluster)
#     DH.setClusterIndices(cluster)
#     DH.setCenterNodes(center_nodes)
#     return cluster,center_nodes,index_of_cluster
def RelocateHub():
   
    n_clusters = DH.getNumberOfClusters()
    cluster = DH.getClusterIndices()
    center_nodes = DH.getCenterNodes()

    for i in range(0,n_clusters):
        hub = center_nodes[i]
        abc = np.copy(cluster[i][0])
        # print("hub -> ",hub)

        clstrList = list(abc)

        center_nodes[i] = int(np.random.choice(clstrList))
        # index = np.where
        # print("center nodes at ",i," -> ",center_nodes[i])
        # cluster[i].append(hub)
        # index_of_cluster = np.where(cluster[i] == center_nodes[i])
        # cluster[i][0][index_of_cluster[1]] = hub
        # np.insert(cluster[i],0,hub)
        clstrList.insert(0,hub)
        # print("holahoalhadrgs")
        index = clstrList.index(center_nodes[i])

        clstrList.pop(index)
        cluster[i] = np.array(clstrList)

        
        # print("indx opfsdfgn ")
        # print(index_of_cluster)
    DH.setClusterIndices(cluster)
    DH.setCenterNodes(center_nodes)

def SwapNodes():
   
    cluster = DH.getClusterIndices()   
    n_clusters = DH.getNumberOfClusters()

    randHub1,randHub2 = np.random.randint(0,n_clusters,2)
    temparr1 = list(cluster[randHub1])
    temparr2 = list(cluster[randHub2])
    print(temparr1)
    print(temparr2)
    temp1 = np.random.choice(list(temparr1))
    temp2 = np.random.choice(list(temparr2))
    ind1 = temparr1.index(temp1)
    ind2 = temparr2.index(temp2)
    temparr1.pop(ind1)
    temparr2.pop(ind2)
    temparr1.insert(ind2,temp2)
    temparr2.insert(ind1,temp1)
    
    # print(temparr1)
    # print(temparr2)
    
    cluster[randHub1] = np.array(temparr1)
    cluster[randHub2] = np.array(temparr2)

    
    DH.setClusterIndices(cluster)       

def ReallocateNode():   
    cluster = DH.getClusterIndices() 
    n_clusters = DH.getNumberOfClusters()
    
    # print(center_nodes)
    # print(cluster)
    randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)

    clusterList1 = list(cluster[randIndex1])
    if len(clusterList1) <=1:
        return
    # print("**\n\n\n\n\n**")
    index = np.random.randint(0,len(clusterList1),1)
    clusterList2 = list(cluster[randIndex2])
    # print(clusterList1)
    # print(clusterList2)
    temp = clusterList1.pop(int(index))
    clusterList2.append(temp)
    cluster[randIndex1] = np.array(clusterList1)
    cluster[randIndex2] = np.array(clusterList2)

    DH.setClusterIndices(cluster)       
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
while limit >40000:
    # indcl = DH.getClusterIndices()
    # limit = DH.getObjectiveResult()
    # centers = DH.getCenterNodes()
    # print(i," --->" ,limit,"---> ",centers,"-->" ,indcl)   
    ReallocateNode()
    calculateCenters()
    calculateFarhestDistance()
    calculatePairCombinations()
    calculatePairObjectives()  
    limit = DH.getObjectiveResult()
    
    time.sleep(0.5)

    i+=1


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


centers = DH.getCenters()
labels = DH.getLabels()
printGraph()    
    
    
    