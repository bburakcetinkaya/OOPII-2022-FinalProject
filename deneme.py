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

from DataHolder import DataHolder
from scipy.spatial import distance
from itertools import combinations
def calculateClusters():
    DH = DataHolder()
    labels = DH.getLabels()
    n_clusters = len(np.unique(labels))
    DH.setNumberOfClusters(n_clusters)
    cluster = [0]*n_clusters
    for i in range(0,(n_clusters)):
        cluster[i] = (np.where(labels==i))
    DH.setClusterIndices(cluster)
    print()
    print("clusters")
    print(cluster)
    print()
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
    print("center nodes")
    print(center_nodes)
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
    print(objective_result)
    
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
    DH = DataHolder()    
    n_clusters = DH.getNumberOfClusters()
    cluster = DH.getClusterIndices()
    center_nodes = DH.getCenterNodes()
    for i in range(0,n_clusters):
        hub = center_nodes[i]
        print("hub -> ",hub)
        center_nodes[i] = np.random.choice(*cluster[i])
        # index = np.where
        print("center nodes at ",i," -> ",center_nodes[i])
        # cluster[i].append(hub)
        # index_of_cluster = np.where(cluster[i] == center_nodes[i])
        # cluster[i][0][index_of_cluster[1]] = hub
        np.insert(cluster[i],0,hub)
        print("holahoalhadrgs")
        arr = cluster[i]
        index = np.where(arr == center_nodes[i])
        print("index ====   " , index) 
        a = np.delete(arr, np.array(index[1]))
        cluster[i] = a
        print(cluster[i])
        
        # print("indx opfsdfgn ")
        # print(index_of_cluster)
    DH.setClusterIndices(cluster)
    DH.setCenterNodes(center_nodes)
def SwapNodes():
    DH = DataHolder()    
    n_clusters = DH.getNumberOfClusters()
    cluster = DH.getClusterIndices()
    
    for i in range(0,n_clusters):
        randHub1,randHub2 = np.random.randint(0,n_clusters,2)
        print("swap nodes")
        print(cluster)
        temp = np.random.choice(cluster[randHub1][0])
        temp2 = np.random.choice(cluster[randHub2][0])
        print("temp = ",randHub1," ",temp)
        print("temp2 = ",randHub2," ",temp2)
        # arr = cluster[randHub1]
        # index = np.where(arr == temp)
        # a = np.delete(arr, np.array(index[0]))
        # cluster[randHub1] = a
        # np.insert(cluster[randHub1],0,temp2)
        
        # arr = cluster[randHub1]
        # index = np.where(arr == temp)
        # a = np.delete(arr, np.array(index[0]))
        # cluster[randHub1] = a
        # print("temp = ",randHub1," ",temp)
        # print("temp2 = ",randHub2," ",temp2)
        # np.insert(cluster[randHub1],0,temp)
        
        print("after")
        print(cluster)
        print()


        
        # print("indx opfsdfgn ")
        # print(index_of_cluster)
    # DH.setClusterIndices(cluster)
    # DH.setCenterNodes(center_nodes)
df = pd.read_csv("C:/Users/piton/Desktop/projects/oop-proj2/data/10.txt",sep=" ",header=None)
df.columns = ["X","Y"]
data = np.array(df)

DH = DataHolder()
X = data
DH.setInitialData(X)
ac = KMeans(n_clusters=3).fit(X)
labels = ac.labels_

DH.setLabels(labels)
calculateClusters()
calculateCenters()
calculateCenterNodes()
##########################################
n_clusters = DH.getNumberOfClusters()  
center_nodes = DH.getCenterNodes()
cluster = DH.getClusterIndices()
# for i in range(0,n_clusters):
#     arr = cluster[i]
#     index = np.where(arr == center_nodes[i])
#     print("index ====   " , index) 
#     a = np.delete(arr, np.array(index[1]))
#     cluster[i] = a
#     print(cluster)
    # cluster[0][i] = a


##########################################
# RelocateHub()

print("before swap")
print(cluster)
SwapNodes()
clusterlast = DH.getClusterIndices()
# print()
# print("after relocate")
# print(clusterlast)
# print()
centernodeslast = DH.getCenterNodes()
# print()
# print("after relocate center nodes")
# print(centernodeslast)
# print()
calculateFarhestDistance()
calculatePairCombinations()
calculatePairObjectives()  


def printGraph():  
    DH = DataHolder()
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
DH = DataHolder()

centers = DH.getCenters()
labels = DH.getLabels()
printGraph()    
    
    
    