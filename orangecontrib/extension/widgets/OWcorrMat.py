from AnyQt.QtWidgets import QFormLayout
from Orange.data import Table, Variable, Domain, ContinuousVariable
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.widgets.widget import Input, Output, AttributeList, Msg
from math import isnan, isinf

from AnyQt.QtWidgets import QTableView, QHeaderView,QSizePolicy
from AnyQt.QtGui import QFont, QBrush, QColor, QStandardItemModel, QStandardItem
from AnyQt.QtCore import Qt, QSize, QItemSelectionModel, QItemSelection, Signal
import numpy as np

from Orange.widgets import widget, gui
from Orange.widgets.settings import \
    Setting, ContextSetting

from Orange.widgets.utils.widgetpreview import WidgetPreview

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt



MAX_COMPONENTS = 100
LINE_NAMES = ["Reconstruction Error"]
LINE_NAMES_TWO = ["component variance", "cumulative variance"]
#widget.OWWidget

BorderRole = next(gui.OrangeUserRole)
BorderColorRole = next(gui.OrangeUserRole)


auto_commit = Setting(True)

class OWcorrMat(widget.OWWidget):

    name = "Correlation heat map"
    description = "Displays correlations between data features."
    icon = "icons/Corrmats.svg"
    priority = 1
    keywords = ["linear discriminant analysis", "linear transformation"]

    class Inputs:

        data = Input("Data", Table)

    def __init__(self):
        super().__init__()
        self.data = None
        self.dataMat = None
        self.dataAttr = None
        self.dataDict = None
        self.train_headers = []
        self.tablemodel = None
        self.legend = np.around(np.arange(start=-1,stop=1.2,step=0.2), decimals=1)

        box = gui.vBox(self.mainArea, "Correlation matrix")
        form = QFormLayout()
        box.layout().addLayout(form)


        self.tablemodel = QStandardItemModel(self)
        view = self.tableview = QTableView(
            editTriggers=QTableView.NoEditTriggers)
        view.setModel(self.tablemodel)
        view.horizontalHeader().hide()
        view.verticalHeader().hide()
        view.horizontalHeader().setMinimumSectionSize(40)
        view.setShowGrid(False)
        view.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)
        box.layout().addWidget(view)

    @staticmethod
    def sizeHint():
        """Initial size"""
        return QSize(933, 600)

    @Inputs.data
    def set_data(self, data):
        self.clear()

        self.data = None
        if isinstance(data, SqlTable):
            if data.approx_len() < AUTO_DL_LIMIT:
                data = Table(data)
            else:
                self.information("Data has been sampled")
                data_sample = data.sample_time(1, no_cache=True)
                data_sample.download_data(2000, partial=True)
                data = Table(data_sample)
        if isinstance(data, Table):
            if not data.domain.attributes:
                self.Error.no_features()
                self.clear_outputs()
                return
            if not data:
                self.Error.no_instances()
                self.clear_outputs()
                return
        self.dataMat = data.X
        self.dataAttr = data.domain.attributes
        self.train_headers = tuple([a.name for a in data.domain.attributes])
        self.dataDict = {}
        counter = 0
        for arg in self.dataAttr:
            self.dataDict[arg] = self.dataMat[:, counter]
            counter = counter + 1

        self.data = data
        self._init_table(len(self.data.domain.attributes))
        self._update_ConfMat()

    def _init_table(self, nfeatures):
        item = self._item(0, 2)
        #item.setData("Predicted", Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.NoItemFlags)
        self._set_item(0, 2, item)

        item = self._item(2, 0)
        #item.setData("Actual", Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        item.setFlags(Qt.NoItemFlags)
        self.tableview.setItemDelegateForColumn(0, gui.VerticalItemDelegate())
        self._set_item(2, 0, item)


        self.tableview.setSpan(0, 2, 1, nfeatures)
        self.tableview.setSpan(2, 0, nfeatures, 1)

        font = self.tablemodel.invisibleRootItem().font()
        bold_font = QFont(font)
        bold_font.setBold(True)

        for i in (0, 1):
            for j in (0, 1):
                item = self._item(i, j)
                item.setFlags(Qt.NoItemFlags)
                self._set_item(i, j, item)


        for p, label in enumerate(self.train_headers):
            for i, j in ((1, p + 2), (p + 2, 1)):
                item = self._item(i, j)
                item.setData(label, Qt.DisplayRole)
                item.setFont(bold_font)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(Qt.ItemIsEnabled)
                if p < len(self.train_headers) - 1:
                    item.setData("br"[j == 1], BorderRole)
                    item.setData(QColor(192, 192, 192), BorderColorRole)
                self._set_item(i, j, item)

        hor_header = self.tableview.horizontalHeader()
        if len(' '.join(self.train_headers)) < 120:
            #hor_header.setSectionResizeMode(QHeaderView.ResizeToContents)
            hor_header.setDefaultSectionSize(40)
        else:
            hor_header.setDefaultSectionSize(40)
        self.tablemodel.setRowCount(nfeatures + 3)
        self.tablemodel.setColumnCount(nfeatures + 6)
        #p = len(self.legend)
        #for i in range(p):
           # item = self._item(i + 3, nfeatures + 6)
           # item.setData(str(self.legend[i]), Qt.DisplayRole)
           # item.setTextAlignment(Qt.AlignCenter)
           # item.setFlags(Qt.NoItemFlags)
           # self._set_item(i + 3, nfeatures + 6, item)

    def _update_ConfMat(self):
        def _isinvalid(x):
            return isnan(x) or isinf(x)

            # Update the displayed confusion matrix
        if self.data is not None:
            a = [self.dataAttr]
            df = pd.DataFrame(self.dataDict, columns=self.dataAttr)
            corrmatrix = df.corr()
            sn.heatmap(corrmatrix, annot=True, cmap="Greens")
            #plt.show()

            cmatrix = corrmatrix.to_numpy()
            cmatrix = cmatrix.round(2)

            n = len(cmatrix)
            diag = np.diag_indices(n)

            colors = cmatrix.astype(np.double)
            colors[diag] = 0

            normalized = cmatrix.astype(np.float)
            formatstr = "{}"
            div = np.array([colors.max()])

            div[div == 0] = 1
            colors /= div
            maxval = normalized[diag].max()
            if maxval > 0:
                colors[diag] = normalized[diag] / maxval

            for i in range(n):
                for j in range(n):
                    val = normalized[i, j]
                    col_val = colors[i, j]
                    item = self._item(i + 2, j + 2)
                    item.setData(
                        "NA" if _isinvalid(val) else formatstr.format(val),
                        Qt.DisplayRole)

                    if col_val < 0:
                        bkcolor = QColor.fromHsl(
                            150, 160,
                            255 if _isinvalid(col_val) else (float(120 - 80 * abs(col_val))))
                    elif col_val >= 0:
                        bkcolor = QColor.fromHsl(
                            150, 160,
                            255 if _isinvalid(col_val) else (float(120 - 80 * col_val)))

                    elif col_val == 0:
                        bkcolor = QColor.fromHsl(
                            [150,160 ][i == j], 200,
                            255 if _isinvalid(col_val) else 255)
                    item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                    item.setData("trbl", BorderRole)
                    item.setToolTip("feature 1: {}\nfeature 2: {}".format(
                        self.train_headers[i], self.train_headers[j]))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self._set_item(i + 2, j + 2, item)
            Colors = self.legend.astype(np.double)
            for j in range(len(self.legend)):
                    val = self.legend[j]
                    col_val = Colors[j]
                    item = self._item(j + 3,len(self.data.domain.attributes) + 4)
                    item.setData(
                        "NA" if _isinvalid(val) else formatstr.format(val),
                        Qt.DisplayRole)

                    if col_val < 0:
                        bkcolor = QColor.fromHsl(
                            150, 160,
                            155 if _isinvalid(col_val) else (float(120 - 80 * abs(col_val))))
                    elif col_val >= 0:
                        bkcolor = QColor.fromHsl(
                           150, 160,
                            255 if _isinvalid(col_val) else (float(120 - 80 * col_val)))

                    elif col_val == 0:
                        bkcolor = QColor.fromHsl(
                            [150,160 ][i == j], 200,
                            255 if _isinvalid(col_val) else 255)
                    item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                    item.setData("trbl", BorderRole)
                    #item.setToolTip("actual: {}\npredicted: {}".format(
                    #    self.train_headers[i], self.train_headers[j]))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self._set_item(j + 3,len(self.data.domain.attributes) + 4, item)

            bold_font = self.tablemodel.invisibleRootItem().font()
            bold_font.setBold(True)

    def _set_selection(self):
        selection = QItemSelection()
        index = self.tableview.model().index
        for row, col in self.selection:
            sel = index(row + 2, col + 2)
            selection.select(sel, sel)
        self.tableview.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

    def _set_item(self, i, j, item):
        self.tablemodel.setItem(i, j, item)

    def _item(self, i, j):
        return self.tablemodel.item(i, j) or QStandardItem()

    def clear(self):
        """Reset the widget, clear controls"""
        self.tablemodel.clear()
        self.train_headers = []

    def send_report(self):
        """Send report"""
        if self.data is not None:
            self.report_table("Correlation matrix", self.tableview)

if __name__ == "__main__":  # pragma: no cover
    #from sklearn.model_selection import KFold

    WidgetPreview(OWcorrMat).run(set_data=Table("housing"))