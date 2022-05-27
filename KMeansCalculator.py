# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:19:55 2022

@author: piton
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from KMeansWindow import Ui_kMeansWindow
import numpy as np
from sklearn.cluster import KMeans

class KMeansCalculator(QtWidgets.QMainWindow,Ui_kMeansWindow):
    def __init__(self,initialData,finalData):
        self.__initialData = initialData
        self.__finalData = finalData
    def getInitialData(self):
        return self.__initialData
