# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:07:19 2022

@author: piton
"""
from PyQt5.QtCore import QObject,pyqtSignal
class SignalSlotCommunicationManager(QObject):
    fileOpened = pyqtSignal()
    kmeansOKClicked = pyqtSignal()
    