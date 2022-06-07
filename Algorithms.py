# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 04:25:56 2022

@author: piton
"""
from DataHolder import DataHolder
import numpy as np

class Algorithms(object):
    def __init__(self):
        # randomSelect = np.random.randint(0,3,1)
        # if randomSelect == 0:
        #     self.RelocateHub()
        # if randomSelect == 1:
        #     self.ReallocateNode()
        # if randomSelect == 2:
        #     self.SwapNodes()
        print("yapıyom")
        self.ReallocateNode()
        print("yaptım")
            
    def RelocateHub(self):
        DH = DataHolder()    
        n_clusters = DH.getNumberOfClusters()
        cluster = DH.getClusterIndices()
        center_nodes = DH.getCenterNodes()
        for i in range(0,n_clusters):
            hub = center_nodes[i]

            center_nodes[i] = np.random.choice(*cluster[i])
            # index = np.where

            np.insert(cluster[i],0,hub)

            arr = cluster[i]
            index = np.where(arr == center_nodes[i])

            a = np.delete(arr, np.array(index[1]))
            cluster[i] = a
            print(cluster[i])

        DH.setClusterIndices(cluster)
        DH.setCenterNodes(center_nodes)

    def SwapNodes(self):
        DH = DataHolder()    
        cluster = DH.getClusterIndices()   
        n_clusters = DH.getNumberOfClusters()

        randHub1,randHub2 = np.random.randint(0,n_clusters,2)
        temparr1 = list(cluster[randHub1][0])
        temparr2 = list(cluster[randHub2][0])
        # print(temparr1)
        # print(temparr2)
        temp1 = np.random.choice(temparr1)
        temp2 = np.random.choice(temparr2)
        ind1 = temparr1.index(temp1)
        ind2 = temparr2.index(temp2)
        temparr1.pop(ind1)
        temparr2.pop(ind2)
        temparr1.insert(ind2,temp2)
        temparr2.insert(ind1,temp1)
        
        cluster[randHub1] = np.array(temparr1)
        cluster[randHub2] = np.array(temparr2)
        
        DH.setClusterIndices(cluster)       

    def ReallocateNode(self):
        DH = DataHolder()    
        cluster = DH.getClusterIndices() 
        n_clusters = DH.getNumberOfClusters()
        randIndex1,randIndex2 = np.random.randint(0,n_clusters,2)
        clusterList1 = list(cluster[randIndex1][0])
        if len(clusterList1) <=1:
            return
        index = np.random.randint(0,len(clusterList1),1)
        clusterList2 = list(cluster[randIndex2][0])
        # print(clusterList1)
        # print(clusterList2)
        temp = clusterList1.pop(int(index))
        clusterList2.append(temp)
        cluster[randIndex1] = np.array(clusterList1)
        cluster[randIndex2] = np.array(clusterList2)
        DH.setClusterIndices(cluster)       
