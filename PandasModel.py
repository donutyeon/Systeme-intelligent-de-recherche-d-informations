from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import pandas as pd

class PandasModel(QtCore.QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None): 
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
    

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        og = self._df.iloc[index.row(),index.column()]
        
        if role == Qt.EditRole:
            self._df.iloc[index.row(),index.column()] = value
            if value == "":
                self._df.iloc[index.row(),index.column()] = og
            return True
            
    def flags(self, index):
        fl = super(self.__class__,self).flags(index)
        fl |= Qt.ItemIsEditable
        fl |= Qt.ItemIsSelectable
        fl |= Qt.ItemIsEnabled
        fl |= Qt.ItemIsDragEnabled
        fl |= Qt.ItemIsDropEnabled

        return fl

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

    # def sort(self, column, order):
    #     colname = self._df.columns.tolist()[column]
    #     self.layoutAboutToBeChanged.emit()
    #     self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
    #     self._df.reset_index(inplace=True, drop=True)
    #     self.layoutChanged.emit()
