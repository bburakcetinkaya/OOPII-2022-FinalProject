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
        self.__best_pair_objectives = []
        self.__best_objective_result = 0
        self.__best_pair_combinations = []
        self.__best_farhest_distances = []
        self.__best_center_nodes = []
        self.__best_cluster_nodes = []
        self.__best_dist = []
        self.__finalData = []
        self.__best_centers = []
        
    def setNumberOfClusters(self,n_clusters):
        self.__n_clusters = n_clusters
    def getNumberOfClusters(self):
        # print("number of clusters = ",self.__n_clusters)
        return self.__n_clusters    
    
    def setPairObjectives(self,pair_objectives):
        self.__pair_objectives = pair_objectives
    def getPairObjectives(self):
        # print("pair objectives = ",self.__pair_objectives)
        return self.__pair_objectives
    def setBestPairObjectives(self,best_pair_objectives):
        self.__best_pair_objectives = best_pair_objectives
    def getBestPairObjectives(self):
        # print("pair objectives = ",self.__pair_objectives)
        return self.__best_pair_objectives
    
    def setObjectiveResult(self,objective_result):
        self.__objective_result = objective_result
    def getObjectiveResult(self):   
        # print("objective result = ",self.__objective_result)
        return self.__objective_result    
    def setBestObjectiveResult(self,best_objective_result):
        self.__best_objective_result = best_objective_result
    def getBestObjectiveResult(self):   
        # print("objective result = ",self.__objective_result)
        return self.__best_objective_result
    
    def setPairCombinations(self,pair_combinations):
        self.__pair_combinations = pair_combinations
    def getPairCombinations(self):
        # print("pair combinations = ",self.__pair_combinations)
        return self.__pair_combinations    
    def setBestPairCombinations(self,best_pair_combinations):
        self.__best_pair_combinations = best_pair_combinations
    def getBestPairCombinations(self):
        # print("pair combinations = ",self.__pair_combinations)
        return self.__best_pair_combinations
    
    def setFarhestHubDistances(self,farhest_distances):
        self.__farhest_distances = farhest_distances
    def getFarhestHubDistances(self):
        # print("farhest distance = ",self.__farhest_distances)
        return self.__farhest_distances    
    def setBestFarhestHubDistances(self,best_farhest_distances):
        self.__best_farhest_distances = best_farhest_distances
    def getBestFarhestHubDistances(self):
        # print("farhest distance = ",self.__farhest_distances)
        return self.__best_farhest_distances
    
    def setCenterNodes(self,center_nodes):
        self.__center_nodes = center_nodes
    def getCenterNodes(self):
        # print("center nodes = ",self.__center_nodes)
        return self.__center_nodes
    def setBestCenterNodes(self,best_center_nodes):
        self.__best_center_nodes = best_center_nodes
    def getBestCenterNodes(self):
        # print("center nodes = ",self.__center_nodes)
        return self.__best_center_nodes
    
    def setClusterNodes(self,cluster_nodes):
        self.__cluster_nodes = cluster_nodes
    def getClusterNodes(self):    
        return self.__cluster_nodes
    def setBestClusterNodes(self,best_cluster_nodes):
        self.__best_cluster_nodes = best_cluster_nodes
    def getBestClusterNodes(self):
        return self.__best_cluster_nodes
    
    def setDistanceMatrix(self,dist):
        self.__dist = dist
    def getDistanceMatrix(self):
        # print("distance = ",self.__dist)
        return self.__dist
    def setBestDistanceMatrix(self,best_dist):
        self.__best_dist = best_dist
    def getBestDistanceMatrix(self):
        # print("distance = ",self.__dist)
        return self.__best_dist
    
    def setFinalData(self,finalData):
        self.__finalData = finalData        
    def getFinalData(self):
        # print("initial data = ",self.__initialData)
        return self.__finalData   
    def setInitialData(self,initialData):
        self.__initialData = initialData        
    def getInitialData(self):
        # print("initial data = ",self.__initialData)
        return self.__initialData 
    
    def setCenters(self,centers):
        self.__centers = centers
    def getCenters(self):
        # print("centers = ",self.__centers)
        return self.__centers    
    def setBestCenters(self,best_centers):
        self.__best_centers = best_centers
    def getBestCenters(self):
        # print("centers = ",self.__centers)
        return self.__best_centers
    
    def setLabels(self,labels):
        self.__labels = labels
    def getLabels(self):
        # print("labels = ",self.__labels)
        return self.__labels