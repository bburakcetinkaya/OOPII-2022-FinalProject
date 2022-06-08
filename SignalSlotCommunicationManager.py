# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:07:19 2022

@author: Burak Çetinkaya
        151220152110
"""
from PyQt5.QtCore import QObject,pyqtSignal
class SignalSlotCommunicationManager(QObject):
    fileOpened = pyqtSignal()
    undoEvent = pyqtSignal()
    redoEvent = pyqtSignal()
    initialClear = pyqtSignal()    
    finalClear = pyqtSignal()    