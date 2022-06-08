# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:54:31 2022

@author: piton
"""

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs

from scipy.spatial import distance
from itertools import combinations

from DataHolder import DataHolder
import numpy as np
import pandas as pd

DH = DataHolder() 
print("Data Read **********************")
df = pd.read_csv("C:/Users/piton/Desktop/projects/oop-proj2/data/10.txt",sep=" ",header=None)
df.columns = ["X","Y"]
data = np.array(df)
DH.setInitialData(data)
km = KMeans(n_clusters=3)
km.fit(data)
labels= km.labels_
kmcenters = km.cluster_centers_
print("km labels = ", labels)
print("km centers = ", kmcenters)
print()

print("Cluster Calculation **********************")
n_clusters = len(np.unique(labels))
DH.setNumberOfClusters(n_clusters)
print("n_clusters = ", n_clusters)
clusterList = []
for i in range(0,(n_clusters)):
    clusterList.append(list(np.where(labels==i))[0])
DH.setClusterIndices(clusterList)
print("cluster list = " ,clusterList)
print()

n_clusters = DH.getNumberOfClusters()
data = DH.getInitialData()
clusterList = DH.getClusterIndices()
print("Centers *******************")
centers = []
for i in range(0,n_clusters):
    x = np.mean(data[[clusterList[i]][0],0])
    y = np.mean(data[[clusterList[i]][0],1])
    centers.append([x,y])
print("centers = " ,centers)
print()

print("Center Nodes *********************")
center_nodes = []
dist = distance.cdist( centers, data, metric="euclidean" )
print("distance = ", dist)
DH.setDistanceMatrix(dist)
cluster_nodes = []
for i in range(0,n_clusters):
        center_nodes.append(np.where(dist[i] == min(dist[i]))[0])
for i in range(0,n_clusters):  
    c = np.array(center_nodes[i])
    d = np.array(clusterList[i])    
    cluster_nodes.append(d[d != c])
print("center nodes = " , center_nodes)        
print()

print("Farhest distance ************************")
farhest_dist = {}
center_nodes_list = []
dist_to_center_node = []
for i in range(0,n_clusters):
    center_nodes_list.append(int(center_nodes[i]))
for i in range(0,n_clusters):  
    # print("data center= ",data[center_nodes_list[i]])
    # print("data = ",data[clusterList[i]])
    a = np.array(data[center_nodes_list[i]])
    a = a.reshape(1,2)
    b = np.array(data[clusterList[i]])
    dist_to_center_node.append(distance.cdist( a, b, metric="euclidean" ))
    

farhest_dist = dict.fromkeys(center_nodes_list,0)
for i in range(0,n_clusters):
        farhest_dist[center_nodes_list[i]] = np.max(dist_to_center_node[i])

print("farhest distances = ",farhest_dist)
print("possible pairs *******************")

pair_combinations = combinations(list(center_nodes),2)
pair_combinations = list(pair_combinations)

DH.setPairCombinations(pair_combinations)
print("Pair combinations =")



# print(center_nodes)
# print(cluster)
# randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
# while randIndex1 == randIndex2:
#     randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
# clusterList1 = list(clusterList[randIndex1])
# clusterList2 = list(clusterList[randIndex2])
# if not len(clusterList1) <=1:
#     print("hello")
#     print()
#     print("cluster nodes befor = ",clusterList)
#     index = int(np.random.randint(0,len(clusterList1),1))
#     temp = clusterList1.pop(int(index))
#     clusterList2.append(temp)
#     clusterList[randIndex1] = np.array(clusterList1)
#     clusterList[randIndex2] = np.array(clusterList2)
#     print("cluster nodes after =" ,clusterList)

# randIndex = int(np.random.randint(0,n_clusters,1))

# cluster = cluster_nodes[randIndex]
# nodeIndex = int(np.random.randint(0,len(cluster),1))
# hub = int(center_nodes[randIndex])
# center_nodes[randIndex] = np.array(cluster[nodeIndex])
# # print("center nodes after = ",center_nodes)
# cluster[nodeIndex] = np.array(hub)
# cluster_nodes[randIndex] = cluster
# print("cluster nodes = ",cluster_nodes)
# print("center nodes = ", center_nodes)



# center_nodes[i] = int(np.random.choice(clstrList))

# clstrList.insert(0,hub)
# index = clstrList.index(center_nodes[i])

# clstrList.pop(index)
# cluster[i] = np.array(clstrList)

print("***************Swap nodes **************")

randHub1 = int(np.random.randint(0,n_clusters,1))
randHub2 = int(np.random.randint(0,n_clusters,1))
while randHub1 == randHub2:
    randHub1 = int(np.random.randint(0,n_clusters,1))

a = cluster_nodes[randHub1]
b = cluster_nodes[randHub2]
randIndex1 = int(np.random.randint(0,len(a),1))
randIndex2 = int(np.random.randint(0,len(b),1))
a[randIndex1],b[randIndex2] = b[randIndex2],a[randIndex1]



# temp = np.random.choice(cluster_nodes[randHub1])
# temp2 = np.random.choice(cluster_nodes[randHub2])
# print("Temp 1",temp," Temp2")
# #swap 1
# arr = cluster_nodes[randHub1]
# index = np.where(arr==temp)
# a = np.delete(arr, np.array(index[0]))
# cluster_nodes[randHub1] = a
# cluster_nodes[randHub1] = np.insert(cluster_nodes[randHub1],0,temp2)

# #swap 2
# arr = cluster_nodes[randHub1]
# index2 = np.where(arr==temp2)
# a = np.delete(arr, np.array(index2[0]))
# cluster_nodes[randHub2] = a
# cluster_nodes[randHub2] = np.insert(cluster_nodes[randHub2],0,temp)

print("After swap node")
print("Cluster Node 1 :", cluster_nodes[randHub1])
print("Cluster Node 2 :", cluster_nodes[randHub2])




