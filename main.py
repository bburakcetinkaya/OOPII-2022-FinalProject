# -*- coding: utf-8 -*-
"""
Created on Mon May 30 02:19:16 2022

@author: Burak Çetinkaya
        151220152110
"""
from MainManager import MainManager
from PyQt5 import QtWidgets
# import time
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainManager()
    win.show()
    # app.aboutToQuit.connect(lambda: {time.sleep(1),win.deleteLater()})
    
    sys.exit(app.exec())