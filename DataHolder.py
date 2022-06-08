# -*- coding: utf-8 -*-
"""
Created on Fri May 27 02:30:26 2022

@author: Burak Çetinkaya
        151220152110
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
        self.__cluster_nodes = []
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
        self.__total_iterations = 0
        self.__result_iteration_number = 0
        self.__executionTime = 0
    def setExecutionTime(self,duration):
        self.__executionTime = duration
    def getExecutionTime(self):
        return self.__executionTime
    def setResultIterationsNumber(self,result_iteration_number):
        self.result_iteration_number = result_iteration_number
    def getResultIterationsNumber(self):
        return self.result_iteration_number
        
    def setTotalIterations(self,total_iterations):
        self.__total_iterations = total_iterations 
    def getTotalIterations(self):
        return self.__total_iterations
    def setNumberOfClusters(self,n_clusters):
        ##print("setNumberOfClusters")
        self.__n_clusters = n_clusters
    def getNumberOfClusters(self):
        # ##print("number of clusters = ",self.__n_clusters)
        ##print("getNumberOfClusters")
        return self.__n_clusters    
        
    def setPairObjectives(self,pair_objectives):
        ##print("setPairObjectives")
        self.__pair_objectives = pair_objectives
    def getPairObjectives(self):
        ##print("getPairObjectives")
        # ##print("pair objectives = ",self.__pair_objectives)
        return self.__pair_objectives
    def setBestPairObjectives(self,best_pair_objectives):
        ##print("setBestPairObjectives")
        self.__best_pair_objectives = best_pair_objectives
    def getBestPairObjectives(self):
        ##print("getBestPairObjectives")
        # ##print("pair objectives = ",self.__pair_objectives)
        return self.__best_pair_objectives
    
    def setObjectiveResult(self,objective_result):
        ##print("setObjectiveResult")
        self.__objective_result = objective_result
        
    def getObjectiveResult(self):
        ##print("getObjectiveResult")
        # ##print("objective result = ",self.__objective_result)
        return self.__objective_result    
    def setBestObjectiveResult(self,best_objective_result):
        ##print("setBestObjectiveResult")
        self.__best_objective_result = best_objective_result
        
    def getBestObjectiveResult(self):   
        ##print("getBestObjectiveResult")
        # ##print("objective result = ",self.__objective_result)
        return self.__best_objective_result
    
    def setPairCombinations(self,pair_combinations):
        ##print("setPairCombinations")
        self.__pair_combinations = pair_combinations
    def getPairCombinations(self):
        ##print("getPairCombinations")
        # ##print("pair combinations = ",self.__pair_combinations)
        return self.__pair_combinations    
    def setBestPairCombinations(self,best_pair_combinations):
        ##print("setBestPairCombinations")
        self.__best_pair_combinations = best_pair_combinations
    def getBestPairCombinations(self):
        ##print("getBestPairCombinations")
        # ##print("pair combinations = ",self.__pair_combinations)
        return self.__best_pair_combinations
    
    def setFarhestHubDistances(self,farhest_distances):
        ##print("setFarhestHubDistances")
        self.__farhest_distances = farhest_distances
    def getFarhestHubDistances(self):
        ##print("getFarhestHubDistances")
        # ##print("farhest distance = ",self.__farhest_distances)
        return self.__farhest_distances    
    def setBestFarhestHubDistances(self,best_farhest_distances):
        ##print("setBestFarhestHubDistances")
        self.__best_farhest_distances = best_farhest_distances
    def getBestFarhestHubDistances(self):
        ##print("getBestFarhestHubDistances")
        # ##print("farhest distance = ",self.__farhest_distances)
        return self.__best_farhest_distances
    
    def setCenterNodes(self,center_nodes):
        ##print("setCenterNodes")
        self.__center_nodes = center_nodes
    def getCenterNodes(self):
        ##print("getCenterNodes")
        # ##print("center nodes = ",self.__center_nodes)
        return self.__center_nodes
    def setBestCenterNodes(self,best_center_nodes):
        ##print("setBestCenterNodes")
        self.__best_center_nodes = best_center_nodes
    def getBestCenterNodes(self):
        ##print("getBestCenterNodes")
        # ##print("center nodes = ",self.__center_nodes)
        return self.__best_center_nodes
    
    def setClusterNodes(self,cluster_nodes):
        #print("setClusterNodes")
        self.__cluster_nodes = cluster_nodes
    def getClusterNodes(self):    
        #print("getClusterNodes")
        return self.__cluster_nodes
    def setBestClusterNodes(self,best_cluster_nodes):
        #print("setBestClusterNodes")
        self.__best_cluster_nodes = best_cluster_nodes
    def getBestClusterNodes(self):
        #print("getBestClusterNodes")
        return self.__best_cluster_nodes
    
    def setDistanceMatrix(self,dist):
        self.__dist = dist
        #print("setDistanceMatrix")
    def getDistanceMatrix(self):
        #print("getDistanceMatrix")
        # #print("distance = ",self.__dist)
        return self.__dist
    def setBestDistanceMatrix(self,best_dist):
        #print("setBestDistanceMatrix")
        self.__best_dist = best_dist
    def getBestDistanceMatrix(self):
        #print("getBestDistanceMatrix")
        # #print("distance = ",self.__dist)
        return self.__best_dist
    
    def setFinalData(self,finalData):
        #print("setFinalData")
        self.__finalData = finalData        
    def getFinalData(self):
        #print("getFinalData")
        # #print("initial data = ",self.__initialData)
        return self.__finalData   
    def setInitialData(self,initialData):
        #print("setInitialData")
        self.__initialData = initialData        
    def getInitialData(self):
        #print("getInitialData")
        # #print("initial data = ",self.__initialData)
        return self.__initialData 
    
    def setCenters(self,centers):
        #print("setCenters")
        self.__centers = centers
    def getCenters(self):
        #print("getCenters")
        # #print("centers = ",self.__centers)
        return self.__centers    
    def setBestCenters(self,best_centers):
        #print("setBestCenters")
        self.__best_centers = best_centers
    def getBestCenters(self):
        #print("getBestCenters")
        # #print("centers = ",self.__centers)
        return self.__best_centers
    
    def setLabels(self,labels):
        #print("setLabels")
        self.__labels = labels
    def getLabels(self):
        #print("getLabels")
        # #print("labels = ",self.__labels)
        return self.__labels