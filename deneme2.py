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
for i in range(0,n_clusters):
        center_nodes.append(np.where(dist[i] == min(dist[i]))[0])
print("center nodes = " , center_nodes)        
print()

print("Farhest distance ************************")
farhest_dist = {}
center_nodes_list = []
dist_to_center_node = []
for i in range(0,n_clusters):
    center_nodes_list.append(int(center_nodes[i]))
for i in range(0,n_clusters):  
    print("data center= ",data[center_nodes_list[i]])
    print("data = ",data[clusterList[i]])
    a = data[center_nodes_list[i]]
    a = a.reshape(1,2)
    b = data[clusterList[i]]
    dist_to_center_node.append(distance.cdist( a, b, metric="euclidean" ))

farhest_dist = dict.fromkeys(center_nodes_list,0)
for i in range(0,n_clusters):
        farhest_dist[center_nodes_list[i]] = (max(dist_to_center_node[i][0]))
# for i in range(0,n_clusters):
#         farhest_dist[center_nodes[i]] = (max(dist[i]))










