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
        self.initialData = []
        self.kMeans_centers = []
        self.kMeans_labels = []
        


    def setInitialData(self,initialData):
        self.initialData = initialData
    def getInitialData(self):
        return self.initialData 
    
    def setKMeansCenters(self,centers):
        self.kMeans_centers = centers
    def getKMeansCenters(self):
        return self.kMeans_centers
    
    def setKMeansLabels(self,labels):
        self.kMeans_labels = labels
    def getKMeansLabels(self):
        return self.kMeans_labels