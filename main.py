import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PandasModel import PandasModel
import sys
import ctypes
import random
import timeit
from collections import Counter
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi,tight_layout=True)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class Ui(QtWidgets.QMainWindow):
        def __init__(self):
                super(Ui, self).__init__()
                uic.loadUi('main.ui', self)
                # self.setWindowIcon(QtGui.QIcon('pick_logo.png'))

                self.button_search_terms=self.findChild(QtWidgets.QPushButton,'bouton_recherche_terme')
                self.button_search_terms.clicked.connect(self.Search_Terms)

                self.button_search_documents=self.findChild(QtWidgets.QPushButton,'bouton_recherche_docu')
                self.button_search_documents.clicked.connect(self.Search_docs)

                self.text_recherche=self.findChild(QtWidgets.QPlainTextEdit,'text_recherche')
                
                self.choix_docu=self.findChild(QtWidgets.QSpinBox,'choix_docu')

                self.table_descripteur = self.findChild(QtWidgets.QTableView,'table_descripteur')
                self.table_descripteur.setAttribute(QtCore.Qt.WA_StyledBackground)  
                self.table_descripteur.setStyleSheet("QTableView {border-radius: 21px; background: #ffc3a3; box-shadow: inset 8px 8px 16px #bf927a, inset -8px -8px 16px #fff4cc;}")
                self.table_descripteur.resizeRowsToContents()
                self.table_descripteur.resizeColumnsToContents()

                self.table_inverse = self.findChild(QtWidgets.QTableView,'table_inverse')
                self.table_inverse.setAttribute(QtCore.Qt.WA_StyledBackground)  
                self.table_inverse.setStyleSheet("QTableView {border-radius: 21px; background: #ffc3a3; box-shadow: inset 8px 8px 16px #bf927a, inset -8px -8px 16px #fff4cc;}")
                self.table_inverse.resizeRowsToContents()
                self.table_inverse.resizeColumnsToContents()

                self.show()
        def Search_Terms(self):
                query = self.text_recherche.toPlainText()
                #preprocess the text
                
                pass
        def Search_docs(self):
                pass

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()



                