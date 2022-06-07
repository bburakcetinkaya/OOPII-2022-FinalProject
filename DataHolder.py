# -*- coding: utf-8 -*-
"""
Created on Fri May 27 02:30:26 2022

@author: piton
"""

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]    

class DataHolder(metaclass=Singleton):
    __metaclass__=Singleton
    def __init__(self):
        self.__initialData = []
        self.__centers = []
        self.__labels = []
        self.__clusterIndices = []
        self.__center_nodes = []
        self.__farhest_distances = []
        self.__pair_combinations = []
        self.__pair_objectives = []
        self.__objective_result = 0
        
    def setNumberOfClusters(self,n_clusters):
        self.__n_clusters = n_clusters
    def getNumberOfClusters(self):
        print("number of clusters = ",self.__n_clusters)
        return self.__n_clusters
    def setPairObjectives(self,pair_objectives):
        self.__pair_objectives = pair_objectives
    def getPairObjectives(self):
        print("pair objectives = ",self.__pair_objectives)
        return self.__pair_objectives
    
    def setObjectiveResult(self,objective_result):
        self.__objective_result = objective_result
    def getObjectiveResult(self):   
        print("objective result = ",self.__objective_result)
        return self.__objective_result
    
    def setPairCombinations(self,pair_combinations):
        self.__pair_combinations = pair_combinations
    def getPairCombinations(self):
        print("pair combinations = ",self.__pair_combinations)
        return self.__pair_combinations
    
    def setFarhestHubDistances(self,farhest_distances):
        self.__farhest_distances = farhest_distances
    def getFarhestHubDistances(self):
        print("farhest distance = ",self.__farhest_distances)
        return self.__farhest_distances
    
    def setCenterNodes(self,center_nodes):
        self.__center_nodes = center_nodes
    def getCenterNodes(self):
        print("center nodes = ",self.__center_nodes)
        return self.__center_nodes
    
    def setDistanceMatrix(self,dist):
        self.__dist = dist
    def getDistanceMatrix(self):
        print("distance = ",self.__dist)
        return self.__dist
    
    def setInitialData(self,initialData):
        self.__initialData = initialData        
    def getInitialData(self):
        print("initial data = ",self.__initialData)
        return self.__initialData 
    
    def setClusterIndices(self,indices):
        self.__clusterIndices = indices
    def getClusterIndices(self):
        print("cluster indices = ",self.__clusterIndices)
        return self.__clusterIndices
        
    def setCenters(self,centers):
        self.__centers = centers
    def getCenters(self):
        print("centers = ",self.__centers)
        return self.__centers
    
    def setLabels(self,labels):
        self.__labels = labels
    def getLabels(self):
        print("labels = ",self.__labels)
        return self.__labels