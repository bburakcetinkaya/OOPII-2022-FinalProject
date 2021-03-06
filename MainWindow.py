# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1167, 683)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setObjectName("mainLayout")
        self.leftLayout = QtWidgets.QVBoxLayout()
        self.leftLayout.setObjectName("leftLayout")
        self.operations_hLayout = QtWidgets.QHBoxLayout()
        self.operations_hLayout.setObjectName("operations_hLayout")
        self.fileButton_open = QtWidgets.QToolButton(self.centralwidget)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/folder-open.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.fileButton_open.setIcon(icon)
        self.fileButton_open.setObjectName("fileButton_open")
        self.operations_hLayout.addWidget(self.fileButton_open)
        self.initialSolution_hLayout = QtWidgets.QGroupBox(self.centralwidget)
        self.initialSolution_hLayout.setEnabled(True)
        self.initialSolution_hLayout.setObjectName("initialSolution_hLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.initialSolution_hLayout)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.initialButton_saveInitialSolution = QtWidgets.QToolButton(self.initialSolution_hLayout)
        self.initialButton_saveInitialSolution.setEnabled(True)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icons/save-file.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.initialButton_saveInitialSolution.setIcon(icon1)
        self.initialButton_saveInitialSolution.setObjectName("initialButton_saveInitialSolution")
        self.horizontalLayout.addWidget(self.initialButton_saveInitialSolution)
        self.initialButton_exportAsInitialSolution = QtWidgets.QToolButton(self.initialSolution_hLayout)
        self.initialButton_exportAsInitialSolution.setEnabled(True)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icons/save-as.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.initialButton_exportAsInitialSolution.setIcon(icon2)
        self.initialButton_exportAsInitialSolution.setObjectName("initialButton_exportAsInitialSolution")
        self.horizontalLayout.addWidget(self.initialButton_exportAsInitialSolution)
        self.initialButton_undo = QtWidgets.QToolButton(self.initialSolution_hLayout)
        self.initialButton_undo.setEnabled(True)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icons/back-arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.initialButton_undo.setIcon(icon3)
        self.initialButton_undo.setObjectName("initialButton_undo")
        self.horizontalLayout.addWidget(self.initialButton_undo)
        self.initialButton_redo = QtWidgets.QToolButton(self.initialSolution_hLayout)
        self.initialButton_redo.setEnabled(True)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("icons/front-arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.initialButton_redo.setIcon(icon4)
        self.initialButton_redo.setObjectName("initialButton_redo")
        self.horizontalLayout.addWidget(self.initialButton_redo)
        self.initialButton_clearInitialSolution = QtWidgets.QToolButton(self.initialSolution_hLayout)
        self.initialButton_clearInitialSolution.setEnabled(True)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icons/clean.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.initialButton_clearInitialSolution.setIcon(icon5)
        self.initialButton_clearInitialSolution.setObjectName("initialButton_clearInitialSolution")
        self.horizontalLayout.addWidget(self.initialButton_clearInitialSolution)
        self.operations_hLayout.addWidget(self.initialSolution_hLayout)
        self.finalSolution_hLayout = QtWidgets.QGroupBox(self.centralwidget)
        self.finalSolution_hLayout.setEnabled(False)
        self.finalSolution_hLayout.setObjectName("finalSolution_hLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.finalSolution_hLayout)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.finalButton_saveFinalSolution = QtWidgets.QToolButton(self.finalSolution_hLayout)
        self.finalButton_saveFinalSolution.setEnabled(False)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("icons/save-file.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.finalButton_saveFinalSolution.setIcon(icon6)
        self.finalButton_saveFinalSolution.setObjectName("finalButton_saveFinalSolution")
        self.horizontalLayout_2.addWidget(self.finalButton_saveFinalSolution)
        self.finalButton_exportAsFinalSolution = QtWidgets.QToolButton(self.finalSolution_hLayout)
        self.finalButton_exportAsFinalSolution.setEnabled(False)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icons/save-as.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.finalButton_exportAsFinalSolution.setIcon(icon7)
        self.finalButton_exportAsFinalSolution.setObjectName("finalButton_exportAsFinalSolution")
        self.horizontalLayout_2.addWidget(self.finalButton_exportAsFinalSolution)
        self.finalButton_undo = QtWidgets.QToolButton(self.finalSolution_hLayout)
        self.finalButton_undo.setIcon(icon3)
        self.finalButton_undo.setObjectName("finalButton_undo")
        self.horizontalLayout_2.addWidget(self.finalButton_undo)
        self.finalButton_redo = QtWidgets.QToolButton(self.finalSolution_hLayout)
        self.finalButton_redo.setIcon(icon4)
        self.finalButton_redo.setObjectName("finalButton_redo")
        self.horizontalLayout_2.addWidget(self.finalButton_redo)
        self.finalButton_clearFinalSolution = QtWidgets.QToolButton(self.finalSolution_hLayout)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("icons/clean.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.finalButton_clearFinalSolution.setIcon(icon8)
        self.finalButton_clearFinalSolution.setObjectName("finalButton_clearFinalSolution")
        self.horizontalLayout_2.addWidget(self.finalButton_clearFinalSolution)
        self.operations_hLayout.addWidget(self.finalSolution_hLayout)
        self.clustering_hLayout = QtWidgets.QGroupBox(self.centralwidget)
        self.clustering_hLayout.setEnabled(False)
        self.clustering_hLayout.setObjectName("clustering_hLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.clustering_hLayout)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.clusteringButton_KMeans = QtWidgets.QToolButton(self.clustering_hLayout)
        self.clusteringButton_KMeans.setInputMethodHints(QtCore.Qt.ImhNone)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("icons/one.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.clusteringButton_KMeans.setIcon(icon9)
        self.clusteringButton_KMeans.setObjectName("clusteringButton_KMeans")
        self.horizontalLayout_3.addWidget(self.clusteringButton_KMeans)
        self.clusteringButton_affinityPropagation = QtWidgets.QToolButton(self.clustering_hLayout)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("icons/two.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.clusteringButton_affinityPropagation.setIcon(icon10)
        self.clusteringButton_affinityPropagation.setObjectName("clusteringButton_affinityPropagation")
        self.horizontalLayout_3.addWidget(self.clusteringButton_affinityPropagation)
        self.clusteringButton_meanShift = QtWidgets.QToolButton(self.clustering_hLayout)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("icons/three.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.clusteringButton_meanShift.setIcon(icon11)
        self.clusteringButton_meanShift.setObjectName("clusteringButton_meanShift")
        self.horizontalLayout_3.addWidget(self.clusteringButton_meanShift)
        self.clusteringButton_spectralClustering = QtWidgets.QToolButton(self.clustering_hLayout)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("icons/four.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.clusteringButton_spectralClustering.setIcon(icon12)
        self.clusteringButton_spectralClustering.setObjectName("clusteringButton_spectralClustering")
        self.horizontalLayout_3.addWidget(self.clusteringButton_spectralClustering)
        self.clusteringButton_hierarchicalClustering = QtWidgets.QToolButton(self.clustering_hLayout)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap("icons/five.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.clusteringButton_hierarchicalClustering.setIcon(icon13)
        self.clusteringButton_hierarchicalClustering.setObjectName("clusteringButton_hierarchicalClustering")
        self.horizontalLayout_3.addWidget(self.clusteringButton_hierarchicalClustering)
        self.clusteringButton_DBSCAN = QtWidgets.QToolButton(self.clustering_hLayout)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap("icons/six.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clusteringButton_DBSCAN.setIcon(icon14)
        self.clusteringButton_DBSCAN.setObjectName("clusteringButton_DBSCAN")
        self.horizontalLayout_3.addWidget(self.clusteringButton_DBSCAN)
        self.operations_hLayout.addWidget(self.clustering_hLayout)
        self.heuristics_hLayout = QtWidgets.QGroupBox(self.centralwidget)
        self.heuristics_hLayout.setEnabled(False)
        self.heuristics_hLayout.setObjectName("heuristics_hLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.heuristics_hLayout)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.heuristicsButton_hillClimbing = QtWidgets.QToolButton(self.heuristics_hLayout)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap("icons/white-one.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.heuristicsButton_hillClimbing.setIcon(icon15)
        self.heuristicsButton_hillClimbing.setObjectName("heuristicsButton_hillClimbing")
        self.horizontalLayout_4.addWidget(self.heuristicsButton_hillClimbing)
        self.heuristicsButton_simulatedAnneling = QtWidgets.QToolButton(self.heuristics_hLayout)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap("icons/white-two.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.heuristicsButton_simulatedAnneling.setIcon(icon16)
        self.heuristicsButton_simulatedAnneling.setObjectName("heuristicsButton_simulatedAnneling")
        self.horizontalLayout_4.addWidget(self.heuristicsButton_simulatedAnneling)
        self.operations_hLayout.addWidget(self.heuristics_hLayout)
        self.operations_hLayout.setStretch(0, 1)
        self.operations_hLayout.setStretch(1, 5)
        self.operations_hLayout.setStretch(2, 5)
        self.operations_hLayout.setStretch(3, 6)
        self.operations_hLayout.setStretch(4, 2)
        self.leftLayout.addLayout(self.operations_hLayout)
        self.solutions_hLayout = QtWidgets.QHBoxLayout()
        self.solutions_hLayout.setObjectName("solutions_hLayout")
        self.initialSolution_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.initialSolution_groupBox.setObjectName("initialSolution_groupBox")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.initialSolution_groupBox)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.initialSolution_graphicsView = QtWidgets.QGraphicsView(self.initialSolution_groupBox)
        self.initialSolution_graphicsView.setObjectName("initialSolution_graphicsView")
        self.horizontalLayout_5.addWidget(self.initialSolution_graphicsView)
        self.solutions_hLayout.addWidget(self.initialSolution_groupBox)
        self.finalSolution_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.finalSolution_groupBox.setObjectName("finalSolution_groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.finalSolution_groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.finalSolution_graphicsView = QtWidgets.QGraphicsView(self.finalSolution_groupBox)
        self.finalSolution_graphicsView.setObjectName("finalSolution_graphicsView")
        self.gridLayout_2.addWidget(self.finalSolution_graphicsView, 0, 0, 1, 1)
        self.solutions_hLayout.addWidget(self.finalSolution_groupBox)
        self.solutions_hLayout.setStretch(0, 1)
        self.solutions_hLayout.setStretch(1, 1)
        self.leftLayout.addLayout(self.solutions_hLayout)
        self.infoPanel_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.infoPanel_groupBox.setObjectName("infoPanel_groupBox")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.infoPanel_groupBox)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.infoPanel_textEdit = QtWidgets.QTextEdit(self.infoPanel_groupBox)
        self.infoPanel_textEdit.setReadOnly(True)
        self.infoPanel_textEdit.setObjectName("infoPanel_textEdit")
        self.horizontalLayout_7.addWidget(self.infoPanel_textEdit)
        self.leftLayout.addWidget(self.infoPanel_groupBox)
        self.leftLayout.setStretch(0, 1)
        self.leftLayout.setStretch(1, 7)
        self.leftLayout.setStretch(2, 3)
        self.mainLayout.addLayout(self.leftLayout)
        self.manualSolution_vLayout = QtWidgets.QVBoxLayout()
        self.manualSolution_vLayout.setObjectName("manualSolution_vLayout")
        self.hubs_nodes_vLayout = QtWidgets.QGroupBox(self.centralwidget)
        self.hubs_nodes_vLayout.setObjectName("hubs_nodes_vLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.hubs_nodes_vLayout)
        self.verticalLayout.setObjectName("verticalLayout")
        self.manualSolution_hubsLabel = QtWidgets.QLabel(self.hubs_nodes_vLayout)
        self.manualSolution_hubsLabel.setObjectName("manualSolution_hubsLabel")
        self.verticalLayout.addWidget(self.manualSolution_hubsLabel)
        self.lineEdit = QtWidgets.QLineEdit(self.hubs_nodes_vLayout)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout.addWidget(self.lineEdit)
        self.manualSolution_nodesLabel = QtWidgets.QLabel(self.hubs_nodes_vLayout)
        self.manualSolution_nodesLabel.setObjectName("manualSolution_nodesLabel")
        self.verticalLayout.addWidget(self.manualSolution_nodesLabel)
        self.textEdit = QtWidgets.QTextEdit(self.hubs_nodes_vLayout)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.manualSolution_runButton = QtWidgets.QPushButton(self.hubs_nodes_vLayout)
        self.manualSolution_runButton.setObjectName("manualSolution_runButton")
        self.verticalLayout.addWidget(self.manualSolution_runButton)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 2)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 5)
        self.verticalLayout.setStretch(4, 1)
        self.manualSolution_vLayout.addWidget(self.hubs_nodes_vLayout)
        self.results_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.results_groupBox.setObjectName("results_groupBox")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.results_groupBox)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.results_textBrowser = QtWidgets.QTextBrowser(self.results_groupBox)
        self.results_textBrowser.setObjectName("results_textBrowser")
        self.horizontalLayout_8.addWidget(self.results_textBrowser)
        self.manualSolution_vLayout.addWidget(self.results_groupBox)
        self.manualSolution_vLayout.setStretch(0, 1)
        self.manualSolution_vLayout.setStretch(1, 7)
        self.mainLayout.addLayout(self.manualSolution_vLayout)
        self.mainLayout.setStretch(0, 3)
        self.mainLayout.setStretch(1, 1)
        self.gridLayout.addLayout(self.mainLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1167, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuExport_As = QtWidgets.QMenu(self.menuFile)
        self.menuExport_As.setObjectName("menuExport_As")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuClear = QtWidgets.QMenu(self.menuEdit)
        self.menuClear.setObjectName("menuClear")
        self.menuUndo = QtWidgets.QMenu(self.menuEdit)
        self.menuUndo.setObjectName("menuUndo")
        self.menuRedo = QtWidgets.QMenu(self.menuEdit)
        self.menuRedo.setObjectName("menuRedo")
        self.menuClustering = QtWidgets.QMenu(self.menubar)
        self.menuClustering.setObjectName("menuClustering")
        self.menuHeuristics = QtWidgets.QMenu(self.menubar)
        self.menuHeuristics.setObjectName("menuHeuristics")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.fileAction_openData = QtWidgets.QAction(MainWindow)
        self.fileAction_openData.setIcon(icon)
        self.fileAction_openData.setObjectName("fileAction_openData")
        self.fileAction_saveInitialSolution = QtWidgets.QAction(MainWindow)
        self.fileAction_saveInitialSolution.setEnabled(False)
        self.fileAction_saveInitialSolution.setIcon(icon1)
        self.fileAction_saveInitialSolution.setObjectName("fileAction_saveInitialSolution")
        self.fileAction_saveFinalSolution = QtWidgets.QAction(MainWindow)
        self.fileAction_saveFinalSolution.setEnabled(False)
        self.fileAction_saveFinalSolution.setIcon(icon1)
        self.fileAction_saveFinalSolution.setObjectName("fileAction_saveFinalSolution")
        self.fileAction_Exit = QtWidgets.QAction(MainWindow)
        self.fileAction_Exit.setEnabled(True)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("icons/exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.fileAction_Exit.setIcon(icon17)
        self.fileAction_Exit.setObjectName("fileAction_Exit")
        self.fileAction_exportAsInitialSolution = QtWidgets.QAction(MainWindow)
        self.fileAction_exportAsInitialSolution.setEnabled(False)
        self.fileAction_exportAsInitialSolution.setIcon(icon2)
        self.fileAction_exportAsInitialSolution.setObjectName("fileAction_exportAsInitialSolution")
        self.fileAction_exportAsFinalSolution = QtWidgets.QAction(MainWindow)
        self.fileAction_exportAsFinalSolution.setEnabled(False)
        self.fileAction_exportAsFinalSolution.setIcon(icon2)
        self.fileAction_exportAsFinalSolution.setObjectName("fileAction_exportAsFinalSolution")
        self.clusteringAction_KMeans = QtWidgets.QAction(MainWindow)
        self.clusteringAction_KMeans.setEnabled(False)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap("icons/one.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clusteringAction_KMeans.setIcon(icon18)
        self.clusteringAction_KMeans.setObjectName("clusteringAction_KMeans")
        self.clusteringAction_affinityPropagation = QtWidgets.QAction(MainWindow)
        self.clusteringAction_affinityPropagation.setEnabled(False)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap("icons/two.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clusteringAction_affinityPropagation.setIcon(icon19)
        self.clusteringAction_affinityPropagation.setObjectName("clusteringAction_affinityPropagation")
        self.clusteringAction_meanShift = QtWidgets.QAction(MainWindow)
        self.clusteringAction_meanShift.setEnabled(False)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap("icons/three.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clusteringAction_meanShift.setIcon(icon20)
        self.clusteringAction_meanShift.setObjectName("clusteringAction_meanShift")
        self.clusteringAction_spectralClustering = QtWidgets.QAction(MainWindow)
        self.clusteringAction_spectralClustering.setEnabled(False)
        icon21 = QtGui.QIcon()
        icon21.addPixmap(QtGui.QPixmap("icons/four.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clusteringAction_spectralClustering.setIcon(icon21)
        self.clusteringAction_spectralClustering.setObjectName("clusteringAction_spectralClustering")
        self.clusteringAction_hierarchicalClustering = QtWidgets.QAction(MainWindow)
        self.clusteringAction_hierarchicalClustering.setEnabled(False)
        icon22 = QtGui.QIcon()
        icon22.addPixmap(QtGui.QPixmap("icons/five.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clusteringAction_hierarchicalClustering.setIcon(icon22)
        self.clusteringAction_hierarchicalClustering.setObjectName("clusteringAction_hierarchicalClustering")
        self.clusteringAction_DBSCAN = QtWidgets.QAction(MainWindow)
        self.clusteringAction_DBSCAN.setEnabled(False)
        self.clusteringAction_DBSCAN.setIcon(icon14)
        self.clusteringAction_DBSCAN.setObjectName("clusteringAction_DBSCAN")
        self.editAction_clearInitialSolution = QtWidgets.QAction(MainWindow)
        self.editAction_clearInitialSolution.setEnabled(False)
        self.editAction_clearInitialSolution.setIcon(icon5)
        self.editAction_clearInitialSolution.setObjectName("editAction_clearInitialSolution")
        self.editAction_clearFinalSolution = QtWidgets.QAction(MainWindow)
        self.editAction_clearFinalSolution.setEnabled(False)
        self.editAction_clearFinalSolution.setIcon(icon5)
        self.editAction_clearFinalSolution.setObjectName("editAction_clearFinalSolution")
        self.heuristicsAction_hillClimbing = QtWidgets.QAction(MainWindow)
        self.heuristicsAction_hillClimbing.setEnabled(False)
        icon23 = QtGui.QIcon()
        icon23.addPixmap(QtGui.QPixmap("icons/white-one.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.heuristicsAction_hillClimbing.setIcon(icon23)
        self.heuristicsAction_hillClimbing.setObjectName("heuristicsAction_hillClimbing")
        self.heuristicsAction_simulatedAnneling = QtWidgets.QAction(MainWindow)
        self.heuristicsAction_simulatedAnneling.setEnabled(False)
        self.heuristicsAction_simulatedAnneling.setIcon(icon16)
        self.heuristicsAction_simulatedAnneling.setObjectName("heuristicsAction_simulatedAnneling")
        self.editAction_undoInitialSolution = QtWidgets.QAction(MainWindow)
        self.editAction_undoInitialSolution.setObjectName("editAction_undoInitialSolution")
        self.editAction_undoFinalSolution = QtWidgets.QAction(MainWindow)
        self.editAction_undoFinalSolution.setObjectName("editAction_undoFinalSolution")
        self.editAction_redoInitialSolution = QtWidgets.QAction(MainWindow)
        self.editAction_redoInitialSolution.setObjectName("editAction_redoInitialSolution")
        self.editAction_redoFinalSolution = QtWidgets.QAction(MainWindow)
        self.editAction_redoFinalSolution.setObjectName("editAction_redoFinalSolution")
        self.menuExport_As.addAction(self.fileAction_exportAsInitialSolution)
        self.menuExport_As.addAction(self.fileAction_exportAsFinalSolution)
        self.menuFile.addAction(self.fileAction_openData)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.fileAction_saveInitialSolution)
        self.menuFile.addAction(self.fileAction_saveFinalSolution)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.menuExport_As.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.fileAction_Exit)
        self.menuClear.addAction(self.editAction_clearInitialSolution)
        self.menuClear.addAction(self.editAction_clearFinalSolution)
        self.menuUndo.addAction(self.editAction_undoInitialSolution)
        self.menuUndo.addAction(self.editAction_undoFinalSolution)
        self.menuRedo.addAction(self.editAction_redoInitialSolution)
        self.menuRedo.addAction(self.editAction_redoFinalSolution)
        self.menuEdit.addAction(self.menuClear.menuAction())
        self.menuEdit.addSeparator()
        self.menuEdit.addAction(self.menuUndo.menuAction())
        self.menuEdit.addAction(self.menuRedo.menuAction())
        self.menuClustering.addAction(self.clusteringAction_KMeans)
        self.menuClustering.addAction(self.clusteringAction_affinityPropagation)
        self.menuClustering.addAction(self.clusteringAction_meanShift)
        self.menuClustering.addAction(self.clusteringAction_spectralClustering)
        self.menuClustering.addAction(self.clusteringAction_hierarchicalClustering)
        self.menuClustering.addAction(self.clusteringAction_DBSCAN)
        self.menuHeuristics.addAction(self.heuristicsAction_hillClimbing)
        self.menuHeuristics.addAction(self.heuristicsAction_simulatedAnneling)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuClustering.menuAction())
        self.menubar.addAction(self.menuHeuristics.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.fileButton_open.setToolTip(_translate("MainWindow", "Open File"))
        self.fileButton_open.setText(_translate("MainWindow", "..."))
        self.initialSolution_hLayout.setTitle(_translate("MainWindow", "Initial Solution"))
        self.initialButton_saveInitialSolution.setToolTip(_translate("MainWindow", "Save"))
        self.initialButton_saveInitialSolution.setText(_translate("MainWindow", "..."))
        self.initialButton_exportAsInitialSolution.setToolTip(_translate("MainWindow", "Save As"))
        self.initialButton_exportAsInitialSolution.setText(_translate("MainWindow", "..."))
        self.initialButton_undo.setToolTip(_translate("MainWindow", "Undo"))
        self.initialButton_undo.setText(_translate("MainWindow", "..."))
        self.initialButton_redo.setToolTip(_translate("MainWindow", "Redo"))
        self.initialButton_redo.setText(_translate("MainWindow", "..."))
        self.initialButton_clearInitialSolution.setToolTip(_translate("MainWindow", "Clear"))
        self.initialButton_clearInitialSolution.setText(_translate("MainWindow", "..."))
        self.finalSolution_hLayout.setTitle(_translate("MainWindow", "Final Solution"))
        self.finalButton_saveFinalSolution.setToolTip(_translate("MainWindow", "Save"))
        self.finalButton_saveFinalSolution.setText(_translate("MainWindow", "..."))
        self.finalButton_exportAsFinalSolution.setToolTip(_translate("MainWindow", "Save As"))
        self.finalButton_exportAsFinalSolution.setText(_translate("MainWindow", "..."))
        self.finalButton_undo.setToolTip(_translate("MainWindow", "Undo"))
        self.finalButton_undo.setText(_translate("MainWindow", "..."))
        self.finalButton_redo.setToolTip(_translate("MainWindow", "Redo"))
        self.finalButton_redo.setText(_translate("MainWindow", "..."))
        self.finalButton_clearFinalSolution.setToolTip(_translate("MainWindow", "Clear"))
        self.finalButton_clearFinalSolution.setText(_translate("MainWindow", "..."))
        self.clustering_hLayout.setTitle(_translate("MainWindow", "Clustering"))
        self.clusteringButton_KMeans.setToolTip(_translate("MainWindow", "K-Means"))
        self.clusteringButton_KMeans.setText(_translate("MainWindow", "K-Means"))
        self.clusteringButton_affinityPropagation.setToolTip(_translate("MainWindow", "Affinity Propagation"))
        self.clusteringButton_affinityPropagation.setText(_translate("MainWindow", "Affinity Propagation"))
        self.clusteringButton_meanShift.setToolTip(_translate("MainWindow", "Mean-Shift"))
        self.clusteringButton_meanShift.setText(_translate("MainWindow", "Mean-Shift"))
        self.clusteringButton_spectralClustering.setToolTip(_translate("MainWindow", "Spectral Clustering"))
        self.clusteringButton_spectralClustering.setText(_translate("MainWindow", "Spectral Clustering"))
        self.clusteringButton_hierarchicalClustering.setToolTip(_translate("MainWindow", "Hierarchical Clustering"))
        self.clusteringButton_hierarchicalClustering.setText(_translate("MainWindow", "Hierarchical Clustering"))
        self.clusteringButton_DBSCAN.setToolTip(_translate("MainWindow", "DBSCAN"))
        self.clusteringButton_DBSCAN.setText(_translate("MainWindow", "DBSCAN"))
        self.heuristics_hLayout.setTitle(_translate("MainWindow", "Heuristics"))
        self.heuristicsButton_hillClimbing.setToolTip(_translate("MainWindow", "Hill Climbing"))
        self.heuristicsButton_hillClimbing.setText(_translate("MainWindow", "..."))
        self.heuristicsButton_simulatedAnneling.setToolTip(_translate("MainWindow", "Simulated Anneling"))
        self.heuristicsButton_simulatedAnneling.setText(_translate("MainWindow", "..."))
        self.initialSolution_groupBox.setTitle(_translate("MainWindow", "Initial Solution"))
        self.finalSolution_groupBox.setTitle(_translate("MainWindow", "Final Solution"))
        self.infoPanel_groupBox.setTitle(_translate("MainWindow", "Info Panel"))
        self.hubs_nodes_vLayout.setTitle(_translate("MainWindow", "Manual Solution"))
        self.manualSolution_hubsLabel.setText(_translate("MainWindow", "Hubs"))
        self.manualSolution_nodesLabel.setText(_translate("MainWindow", "Nodes"))
        self.manualSolution_runButton.setText(_translate("MainWindow", "RUN"))
        self.results_groupBox.setTitle(_translate("MainWindow", "Results"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuExport_As.setTitle(_translate("MainWindow", "Export As"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuClear.setTitle(_translate("MainWindow", "Clear"))
        self.menuUndo.setTitle(_translate("MainWindow", "Undo"))
        self.menuRedo.setTitle(_translate("MainWindow", "Redo"))
        self.menuClustering.setTitle(_translate("MainWindow", "Clustering"))
        self.menuHeuristics.setTitle(_translate("MainWindow", "Heuristics"))
        self.fileAction_openData.setText(_translate("MainWindow", "Open Data"))
        self.fileAction_openData.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.fileAction_saveInitialSolution.setText(_translate("MainWindow", "Save Initial Solution"))
        self.fileAction_saveInitialSolution.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.fileAction_saveFinalSolution.setText(_translate("MainWindow", "Save Final Solution"))
        self.fileAction_Exit.setText(_translate("MainWindow", "Exit"))
        self.fileAction_Exit.setShortcut(_translate("MainWindow", "Shift+F4"))
        self.fileAction_exportAsInitialSolution.setText(_translate("MainWindow", "Initial Solution"))
        self.fileAction_exportAsFinalSolution.setText(_translate("MainWindow", "Final Solution"))
        self.clusteringAction_KMeans.setText(_translate("MainWindow", "K-Means"))
        self.clusteringAction_KMeans.setShortcut(_translate("MainWindow", "Ctrl++"))
        self.clusteringAction_affinityPropagation.setText(_translate("MainWindow", "Affinity Propagation"))
        self.clusteringAction_affinityPropagation.setShortcut(_translate("MainWindow", "Ctrl+-"))
        self.clusteringAction_meanShift.setText(_translate("MainWindow", "Mean-Shift"))
        self.clusteringAction_spectralClustering.setText(_translate("MainWindow", "Spectral Clustering"))
        self.clusteringAction_hierarchicalClustering.setText(_translate("MainWindow", "Hierarchical Clustering"))
        self.clusteringAction_DBSCAN.setText(_translate("MainWindow", "DBSCAN"))
        self.editAction_clearInitialSolution.setText(_translate("MainWindow", "Initial Solution"))
        self.editAction_clearInitialSolution.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.editAction_clearFinalSolution.setText(_translate("MainWindow", "Final Solution"))
        self.editAction_clearFinalSolution.setShortcut(_translate("MainWindow", "Ctrl+Shift+L"))
        self.heuristicsAction_hillClimbing.setText(_translate("MainWindow", "Hill Climbing"))
        self.heuristicsAction_hillClimbing.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.heuristicsAction_simulatedAnneling.setText(_translate("MainWindow", "Simulated Anneling"))
        self.heuristicsAction_simulatedAnneling.setShortcut(_translate("MainWindow", "Ctrl+B"))
        self.editAction_undoInitialSolution.setText(_translate("MainWindow", "Initial Solution"))
        self.editAction_undoFinalSolution.setText(_translate("MainWindow", "Final Solution"))
        self.editAction_redoInitialSolution.setText(_translate("MainWindow", "Initial Solution"))
        self.editAction_redoFinalSolution.setText(_translate("MainWindow", "Final Solution"))
