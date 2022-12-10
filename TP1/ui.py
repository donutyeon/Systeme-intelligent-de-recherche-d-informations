from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QVBoxLayout, QFileDialog, QMessageBox
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
import numpy as np
import os
import sys
import warnings

from idk import *
warnings.filterwarnings("ignore")



class Appdemo(QtWidgets.QMainWindow):
    def __init__(self):
    
        super().__init__()
        uic.loadUi('main_window.ui',self)

        self.query_field=self.findChild(QtWidgets.QTextEdit,'request')
        self.search_button=self.findChild(QtWidgets.QPushButton,'search')
        self.search_button.clicked.connect(self.search_docus)
        self.table_view=self.findChild(QtWidgets.QTableView,'table')

        self.termes_par_docu=self.findChild(QtWidgets.QRadioButton,'terme_par_docu')
    
    def search_docus(self):
        pass



            



        

    
            

    

   
if __name__ == "__main__":
    
    
    app = QApplication(sys.argv)
    demo =Appdemo()
    demo.show()
    sys.exit(app.exec_())
        