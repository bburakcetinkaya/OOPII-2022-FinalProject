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

from scipy.spatial import distance
from itertools import combinations

# # df = pd.read_csv("C:/Users/piton/Desktop/projects/oop-proj2/data/10.txt",sep=" ",header=None)
# # df.columns = ["X","Y"]
# X = np.array(df)
data, labels_true = make_blobs(n_samples=10, cluster_std=0.4, random_state=0)   
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
X = data
ac = KMeans().fit(X)
labels = ac.labels_
n_clusters = len(np.unique(ac.labels_))
print("labels:")
print(labels)
print()
print("number of clusters: ",n_clusters)
    
cluster = [0]*n_clusters
for i in range(0,(n_clusters)):
    cluster[i] = np.array((np.where(labels==i)))
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




def RelocateHub():
    for i in range(0,n_clusters):
        # listeden center node lar silinecek öyle çalışır bence :) kolay gelsin koçum benim beeeee 
        hub = center_nodes[i]
        print("hub -> ",hub)
        center_nodes[i] = np.random.choice(*cluster[i])
        print("center nodes at ",i," -> ",center_nodes[i])
        index_of_cluster = np.where(cluster[i] == center_nodes[i])
        cluster[i][0][index_of_cluster[1]] = hub
    
    
    
RelocateHub()  

def printGraph(__data,__centers,__labels):  
    initialSolution_figure = plt.figure()
    initialSolution_canvas = FigureCanvas(initialSolution_figure)
            
    # self.initialSolution_figure.clear()
    ploting = initialSolution_figure.add_subplot(111)
    
    if len(__data):
        ploting.scatter(__data[:,0], __data[:,1],color="k",s=20) 
        # for i in range(len(self.__data)):
        #     plt.annotate(str(i),(self.__data[:,0], self.__data[:,1]))
        # print("data")
    
    if len(__labels):
        ploting.scatter(__data[:,0], __data[:,1],c = __labels,s = 20,cmap = 'rainbow')
        # for i in range(len(self.__data)):
        #     plt.annotate(str(i),(self.__data[:,0], self.__data[:,1]))
        # print("lbl")
    if len(__centers):
        ploting.scatter(np.array(__centers)[:, 0],np.array(__centers)[:, 1],c = "red",s = 100, marker="x",alpha = 1,linewidth=1)
        # print("center")

printGraph(X,centers,labels)    
    
    
    