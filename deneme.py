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
import numpy as np
import pandas as pd

from scipy.spatial import distance
from itertools import combinations

df = pd.read_csv("C:/Users/piton/Desktop/projects/oop-proj2/data/10.txt",sep=" ",header=None)
df.columns = ["X","Y"]
X = np.array(df)
# n_clusters=3
# clustering = KMeans(n_clusters).fit(X)
# labels = clustering.labels_
# cluster = [0]*n_clusters
# for i in range(0,(n_clusters)):
#     cluster[i] = (np.where(labels==i))
# for i in range(0,(n_clusters)):
#     print("cluster ",i,"-->",*cluster[i])
# center_nodes = []  
# centers = clustering.cluster_centers_ 
# dist = distance.cdist( centers, X, metric="euclidean" )
# for i in range(0,n_clusters):
#     center_nodes.append(np.where(dist[i] == min(dist[i])))

# center_nodes = np.array(center_nodes)
# center_nodes = center_nodes.reshape(1,n_clusters)
# center_nodes = list(*center_nodes)    
# print("Cluster center nodes -->",*center_nodes)

# farhest_dist = {}
# farhest_dist = dict.fromkeys(center_nodes, 0)
# for i in range(0,n_clusters):
#     farhest_dist[center_nodes[i]] = (max(dist[i]))


# pair_combinations = combinations(list(center_nodes),2)
# pair_combinations = list(pair_combinations)


# pair_objectives = np.zeros(len(pair_combinations))
# for i in range(0,len(pair_combinations)):
#     cluster_i = pair_combinations[i][0]
#     cluster_j = pair_combinations[i][1]
#     dihi = farhest_dist[cluster_i]
    
#     p1 = X[cluster_i]
#     p1 = p1.reshape(1,2)
#     p2 = X[cluster_j]
#     p2 = p2.reshape(1,2)
#     dhihj = distance.cdist(p1 ,p2,metric="euclidean")
#     djhj = farhest_dist[cluster_j]
#     objij = dihi+0.75*dhihj+djhj
#     pair_objectives[i] = (objij)
    
#     objective_result = max(pair_objectives)

# print(pair_objectives)
# print()
# print(objective_result)
    

# a = {}
# for i in range(0,n_clusters):
#      a = dict(map(center_nodes[i],*farhest_dist[i]))


# ap = AffinityPropagation(random_state=15).fit(X)
# labels = ap.labels_
# print("labels:")
# print(labels)

# n_clusters = len(np.unique(ap.labels_))
# print("number of clusters : "  ,n_clusters)

# cluster = [0]*n_clusters
# for i in range(0,(n_clusters)):
#     cluster[i] = (np.where(labels==i))
# for i in range(0,(n_clusters)):
#     print("cluster ",i,"-->",*cluster[i])
    
# centers = [0]*n_clusters
# for i in range(0,n_clusters):
#     x = np.mean(X[np.where(labels==i)][:,0])
#     y = np.mean(X[np.where(labels==i)][:,1])
#     centers[i] = [x,y]

# print("centers: ")
# print(centers)

# center_nodes = []  
# dist = distance.cdist(centers, X, metric="euclidean" )
# for i in range(0,n_clusters):
#     center_nodes.append(np.where(dist[i] == min(dist[i])))

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
    
#     p1 = X[cluster_i]
#     p1 = p1.reshape(1,2)
#     p2 = X[cluster_j]
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


# n_clusters = len(np.unique(ap.labels_))
# center_nodes = []  
# centers = ap.cluster_centers_ 
# dist = distance.cdist( centers, X, metric="euclidean" )
# for i in range(0,n_clusters):
#     center_nodes.append(np.where(dist[i] == min(dist[i])))

# center_nodes = np.array(center_nodes)
# center_nodes = center_nodes.reshape(1,n_clusters)
# center_nodes = list(*center_nodes)    


# farhest_dist = {}
# farhest_dist = dict.fromkeys(center_nodes, 0)
# for i in range(0,n_clusters):
#     farhest_dist[center_nodes[i]] = (max(dist[i]))            


# pair_combinations = combinations(list(center_nodes),2)
# pair_combinations = list(pair_combinations)


# pair_objectives = np.zeros(len(pair_combinations))
# for i in range(0,len(pair_combinations)):
#     cluster_i = pair_combinations[i][0]
#     cluster_j = pair_combinations[i][1]
#     dihi = farhest_dist[cluster_i]
    
#     p1 = X[cluster_i]
#     p1 = p1.reshape(1,2)
#     p2 = X[cluster_j]
#     p2 = p2.reshape(1,2)
#     dhihj = distance.cdist(p1 ,p2,metric="euclidean")
#     djhj = farhest_dist[cluster_j]
#     objij = dihi+0.75*dhihj+djhj
#     pair_objectives[i] = (objij)


# objective_result = max(pair_objectives)
# print(objective_result)

ac = AgglomerativeClustering().fit(X)
labels = ac.labels_
n_clusters = ac.n_clusters_
print("labels:")
print(labels)
print()
print("number of clusters: ",n_clusters)
    
cluster = [0]*n_clusters
for i in range(0,(n_clusters)):
    cluster[i] = (np.where(labels==i))
for i in range(0,(n_clusters)):
    print("cluster ",i,"-->",*cluster[i])
    
centers = [0]*n_clusters
for i in range(0,n_clusters):
    x = np.mean(X[np.where(labels==i)][:,0])
    y = np.mean(X[np.where(labels==i)][:,1])
    centers[i] = [x,y]

print("centers: ")
print(centers)

center_nodes = []  
dist = distance.cdist(centers, X, metric="euclidean" )
for i in range(0,n_clusters):
    center_nodes.append(np.where(dist[i] == min(dist[i])))

center_nodes = np.array(center_nodes)
center_nodes = center_nodes.reshape(1,n_clusters)
center_nodes = list(*center_nodes)    
print("Cluster center nodes -->",*center_nodes)

farhest_dist = {}
farhest_dist = dict.fromkeys(center_nodes, 0)
for i in range(0,n_clusters):
    farhest_dist[center_nodes[i]] = (max(dist[i]))

print("****Farhest hub distances****")
print(farhest_dist)

pair_combinations = combinations(list(center_nodes),2)
pair_combinations = list(pair_combinations)
print("possible pair combinations --> ",pair_combinations)


pair_objectives = np.zeros(len(pair_combinations))
for i in range(0,len(pair_combinations)):
    cluster_i = pair_combinations[i][0]
    cluster_j = pair_combinations[i][1]
    dihi = farhest_dist[cluster_i]
    
    p1 = X[cluster_i]
    p1 = p1.reshape(1,2)
    p2 = X[cluster_j]
    p2 = p2.reshape(1,2)
    dhihj = distance.cdist(p1 ,p2,metric="euclidean")
    djhj = farhest_dist[cluster_j]
    objij = dihi+0.75*dhihj+djhj
    pair_objectives[i] = (objij)
    
    objective_result = max(pair_objectives)

print("****Pair objectives****")
print(pair_objectives)
print()
print("Objective result --> " ,objective_result)