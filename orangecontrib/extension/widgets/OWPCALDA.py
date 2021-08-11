
import numpy
from orangewidget.gui import tabWidget, createTabPage
from AnyQt.QtWidgets import QFormLayout
from Orange.data import Table, Variable, Domain, ContinuousVariable, StringVariable
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from orangecontrib.extension.utils.Projection.lda import LDA, LDAtestTransform
from Orange.widgets.widget import Input, Output, AttributeList, Msg
from orangecontrib.extension.utils import scattergraph
from Orange.widgets.utils.itemmodels import DomainModel
from math import isnan, isinf
from itertools import chain
import unicodedata
from AnyQt.QtWidgets import QTableView, QHeaderView,QSizePolicy
from AnyQt.QtGui import QFont, QBrush, QColor, QStandardItemModel, QStandardItem
from AnyQt.QtCore import Qt, QSize, QItemSelectionModel, QItemSelection, Signal
import numpy as np
import sklearn.metrics as skl_metrics
from sklearn.model_selection import cross_val_score
from Orange.data.util import get_unique_names
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import Orange
import Orange.evaluation
from Orange.widgets import widget, gui
from Orange.widgets.settings import \
    Setting, ContextSetting
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import time
from joblib import Parallel, delayed, parallel_backend
from Orange.widgets.utils.slidergraph import SliderGraph



MAX_COMPONENTS = 100
LINE_NAMES = ["Reconstruction Error"]
LINE_NAMES_TWO = ["component variance", "cumulative variance"]


BorderRole = next(gui.OrangeUserRole)
BorderColorRole = next(gui.OrangeUserRole)
learner_name = "LDA"



def confusion_matrix(data, pred):
    """
    Compute confusion matrix

    Args:
        res (Orange.evaluation.Results): evaluation results
        index (int): model index

    Returns: Confusion matrix
    """

    labels = np.arange(len(data.domain.class_var.values))
    if not data.Y.size:

        return np.zeros((len(labels), len(labels)))
    else:
        return skl_metrics.confusion_matrix(
            y_true=data.Y, y_pred=pred, labels=labels)


class OWPCALDA(widget.OWWidget):

    name = "PCA-LDA"
    description = "Performs a linear discriminant analysis of pca score data and " \
                  "displays a lda score plot, a confusion matrix and a cross " \
                  "validation error-plot"
    icon = "icons/PCALDA.svg"
    priority = 1
    keywords = ["linear discriminant analysis", "linear transformation"]

    class Inputs:

        train_data = Input("Data", Table)
        test_data = Input("Test Data", Table)

    class Outputs:

        lda = Output("LDA", LDA, dynamic=False)

    class Error(widget.OWWidget.Error):
        sparse_train_Matrix = Msg("Train Data contains NaN")
        sparse_test_Matrix = Msg("Test Data contains NaN")
        invalid_values = Msg("Class Data contains NaN")
        empty_input = widget.Msg("Empty result on input. Nothing to display.")

    quantities = ["Number of instances",
                  "Proportion of predicted",
                  "Proportion of actual"]


    CVE = Setting(10)
    selected_quantity = Setting(0)

    auto_commit = Setting(True)
    class_box = Setting(True)
    legend_box = Setting(False)
    axis_labels = Setting(10)
    ncomponents = Setting(4)
    maxp = Setting(20)
    testdata_box = Setting(False)
    testdata_classes_box = Setting(False)

    selection = ContextSetting(set())
    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)

    TestOnTrain, TestOnTest, LeaveOneOut, KFold, StratifiedKFold = 0, 1, 2, 3, 4
    NFolds = [2, 3, 5, 10]
    sNFolds = [2, 3, 5, 10]

    resampling = Setting(0, schema_only=True)
    n_folds = Setting(3)
    sn_folds = Setting(3)

    xy_changed_manually = Signal(Variable, Variable)
    common_options = dict(
        labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True,
        contentsLength=14
    )

    def __init__(self):

        super().__init__()
        self.parallel = Parallel(n_jobs=1, prefer="threads", pre_dispatch='2*n_jobs')
        self.train_data = None
        self.test_data = None
        self.train_classes = None
        self.test_classes = None
        self.plot = None
        self.plotTwo = None
        self.tablemodel = None
        self.train_headers = []
        self.test_headers = []
        self._CVE = None
        self._lda = None
        #self._ldaCV = None
        self._transformed = None
        self._transformedCV = None
        self.train_pred = None
        self.classes = None
        self.CV_test_pred = None
        self.test_pred = None
        self.zaehler = 0

        self.domainIndexes = {}
        self.datalabel = None
        self.train_datalabel = None
        self.test_datalabel = None
        self.PlotStyle = None
        self.testlabel = None
        self.train_class_values = None
        self.test_class_values = None


        self.SYMBOLBRUSH = [(0, 204, 204, 180), (51, 255, 51, 180), (255, 51, 51, 180), (0, 128, 0, 180),  \
                       (195, 46, 212, 180), (250, 194, 5, 180), (55, 55, 55, 180), (0, 114, 189, 180), (217, 83, 25, 180), (237, 177, 32, 180), \
                       (126, 47, 142, 180), (119, 172, 180)]

        self.SYMBOLPEN = [(0, 204, 204, 255), (51, 255, 51, 255), (255, 51, 51, 255), (0, 128, 0, 255),  \
                       (195, 46, 212, 255), (250, 194, 5, 255), (55, 55, 55, 255), (0, 114, 189, 255), (217, 83, 25, 255), (237, 177, 32, 255), \
                       (126, 47, 142, 255), (119, 172, 255)]


        self._init_projector()

        box = gui.vBox(self.controlArea, "Discriminant function selection")
        form = QFormLayout()
        box.layout().addLayout(form)


        dmod = DomainModel
        self.xy_model = DomainModel(dmod.MIXED, valid_types=ContinuousVariable)
        self.cb_attr_x = gui.comboBox(
            box, self, "attr_x", label=None,
            callback=self.set_attr_from_combo,
            model=self.xy_model, **self.common_options,
            searchable=True)
        self.cb_attr_y = gui.comboBox(
            box, self, "attr_y", label=None,
            callback=self.set_attr_from_combo,
            model=self.xy_model, **self.common_options,
            searchable=True)

        form.addRow("Axis x:", self.cb_attr_x)
        form.addRow("Axis y:", self.cb_attr_y)

        class_box = gui.vBox(self.controlArea, "Scatter Plot options")

        self.classb = gui.checkBox(class_box,
            self, value="class_box", label="Color by class",
            callback=self._update_class_box, tooltip="Datapoints get colored by class, when checked")

        self.legendb = gui.checkBox(class_box,
            self, value="legend_box", label="Show legend",
            callback=self._update_legend_box, tooltip=None)

        self.testdatab = gui.checkBox(class_box,
            self, value="testdata_box", label="Show test data",
            callback=self._update_testdata_box, tooltip=None)

        self.testdatabc = gui.checkBox(class_box,
            self, value="testdata_classes_box", label="Hide test data classes",
            callback=self._update_testdata_classes_box, tooltip=None)

        box = gui.vBox(self.controlArea, "Confusion matrix options")
        form = QFormLayout()
        box.layout().addLayout(form)

        rbox = gui.radioButtons(
            box, self, "resampling", callback=self._param_changed)
        gui.appendRadioButton(rbox, "Test on train data")
        gui.appendRadioButton(rbox, "Test on test data")
        gui.appendRadioButton(rbox, "Leave one out")
        gui.appendRadioButton(rbox, "Cross validation")
        ibox = gui.indentedBox(rbox)
        gui.comboBox(
            ibox, self, "n_folds", label="Number of folds: ",
            items=self.NFolds,
            orientation=Qt.Horizontal, callback=self.kfold_changed)
        gui.appendRadioButton(rbox, "Stratified cross validation")
        ibox = gui.indentedBox(rbox)
        gui.comboBox(
            ibox, self, "sn_folds", label="Number of folds: ",
            items=[str(x) for x in self.sNFolds],
            orientation=Qt.Horizontal, callback=self.skfold_changed)
        form.addRow("Evaluation mode:", rbox)

        box = gui.vBox(self.controlArea, "CVE-Plot Component Selection")
        form = QFormLayout()
        box.layout().addLayout(form)

        self.components_spin = gui.spin(
            box, self, "ncomponents", 1, MAX_COMPONENTS,
            callback=self._update_selection_component_spin,
            keyboardTracking=False)

        form.addRow("Components:", self.components_spin)

        self.options_box = gui.vBox(self.controlArea, "CVE-Plot Options")
        form = QFormLayout()
        box.layout().addLayout(form)

        self.maxp_spin = gui.spin(
            self.options_box, self, "maxp", 1, MAX_COMPONENTS,
            label="Show only first", callback=self._setup_plot_Two
        )

        self.controlArea.layout().addStretch()

        gui.auto_apply(self.controlArea, self, "auto_commit")
        tabs = tabWidget(self.mainArea)

        boxScatter = gui.vBox(self.mainArea, "Scatterplot")
        formScatter = QFormLayout()
        boxScatter.layout().addLayout(formScatter)

        self.plot = scattergraph.ScatterGraph(callback=None)
        boxScatter.layout().addWidget(self.plot)
        tab = createTabPage(tabs, "Scatterplot")
        tab.layout().addWidget(boxScatter)

        boxConfus = gui.vBox(self.mainArea, "Confusion matrix")
        formConfus = QFormLayout()
        boxConfus.layout().addLayout(formConfus)

        sbox = gui.hBox(boxConfus)
        gui.rubber(sbox)
        gui.comboBox(sbox, self, "selected_quantity",
                     items=self.quantities, label="Show: ",
                     orientation=Qt.Horizontal, callback=self._update_ConfMat)

        self.tablemodel = QStandardItemModel(self)
        view = self.tableview = QTableView(
            editTriggers=QTableView.NoEditTriggers)
        view.setModel(self.tablemodel)
        view.horizontalHeader().hide()
        view.verticalHeader().hide()
        view.horizontalHeader().setMinimumSectionSize(110)
        view.setShowGrid(False)
        view.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)
        boxConfus.layout().addWidget(view)

        tab = createTabPage(tabs, "Confusion matrix")
        tab.layout().addWidget(boxConfus)

        boxCVE = gui.vBox(self.mainArea, "Scatterplot")
        formCVE = QFormLayout()
        boxCVE.layout().addLayout(formCVE)

        self.plotTwo = SliderGraph(
            "Components", "Cross Validation Error",
            callback=self._on_cut_changed)
        boxCVE.layout().addWidget(self.plotTwo)
        tab = createTabPage(tabs, "CVE-plot")
        tab.layout().addWidget(boxCVE)

#widget properties
    @staticmethod
    def sizeHint():
        """Initial size"""
        return QSize(933, 600)

    def _prepare_data(self):
        indices = self.tableview.selectedIndexes()
        indices = {(ind.row() - 2, ind.column() - 2) for ind in indices}
        actual = self.train_data.Y
        predicted = self.train_pred
        selected = [i for i, t in enumerate(zip(actual, predicted))
                    if t in indices]

        extra = []
        class_var = self.train_data.domain.class_var
        metas = self.train_data.domain.metas
        attrs = self.train_data.domain.attributes
        names = [var.name for var in chain(metas, [class_var], attrs)]
        domain = Orange.data.Domain(self.train_data.domain.attributes,
                                    self.train_data.domain.class_vars,
                                    metas)
        data = self.train_data.transform(domain)
        if extra:
            data.metas[:, len(self.train_data.domain.metas):] = \
                np.hstack(tuple(extra))
        data.name = learner_name

        if selected:
            annotated_data = create_annotated_table(data, selected)
            data = data[selected]
        else:
            annotated_data = create_annotated_table(data, [])
            data = None

        return data, annotated_data

    def commit(self):

        """Output data instances corresponding to selected cells"""
        self.Outputs.lda.send(self._lda_projector)

    def send_report(self):
        """Send report"""
        """Send report"""
        if self.train_data is None:
            return
        if self.train_pred is not None:
            self.report_plot("Score plot", self.plot)
            self.report_table("Confusion matrix", self.tableview)
            self.report_plot("Cross validation error plot", self.plotTwo)

#Data handling
    @Inputs.train_data
    def set_train_data(self, data):

        self.clear()
        self.train_data = None

        if (data == None):
            self.Error.empty_input()
            return

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

        self._init_projector()

        self.train_data = data
        self.train_datalabel = self.train_data.Y
        self.classes = numpy.unique(self.train_data.Y)
        if self.train_data.domain.class_var:
            self.train_classes = {int(self.classes[i]) : self.train_data.domain.class_var.values[i] for i in numpy.arange(0,len(self.train_data.domain.class_var.values))}
        self._lda, self._transformed, self.train_pred, self.train_class_values = self._fit(data)

        self.train_headers = self.train_class_values + \
                       (unicodedata.lookup("N-ARY SUMMATION"), )
        self.init_attr_values()
        self._setup_plot(self.attr_x, self.attr_y)
        self._init_table(len(self.train_class_values))
        self._update_ConfMat()

    @Inputs.test_data
    def set_test_data(self, data):

        if (data == None):
            self.Error.empty_input()
            return
        self.test_data = None
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

        self.test_data = data
        self.test_datalabel = self.test_data.Y
        b = numpy.unique(self.test_data.Y)
        if self.test_data.domain.class_var:
            self.test_classes = {int(b[i]) : self.test_data.domain.class_var.values[i] for i in numpy.arange(0,len(self.test_data.domain.class_var.values))}

        if data is not None:
            if np.any(np.isnan(self.test_data.X)):
                self.Error.sparse_test_Matrix()
            elif self._lda is None:
                time.sleep(3)
            elif self._lda is not None:
                self.test_pred = self._lda.proj.predict(self.test_data.X)
                self.test_class_values = data.domain.class_var.values
                self._testdata_transformed = self.testdata_transform(self.test_data, self._lda)
        if self.test_pred is not None:
            nan_values = False
            if np.any(np.isnan(self.train_data.Y)) or \
                    np.any(np.isnan(self.test_pred)):

                nan_values = True
                pred = data = None

            self.test_headers = self.test_class_values + \
                       (unicodedata.lookup("N-ARY SUMMATION"), )

            self.Error.invalid_values(shown=nan_values)
            if self.testdata_box:
                self.init_attr_values()
                self._setup_plot(self.attr_x, self.attr_y)
            self._update_ConfMat()
            self._update_plot_two()
            self.unconditional_commit()

    def _fit(self, data=None, testset=None):

        self.clear()
        if data is None:
            return
        lda = self._lda_projector(data[:])

        if data is not None:
            if np.any(np.isnan(data.X)):
                self.Error.sparse_train_Matrix()
            else:
                pred = lda.proj.predict(data.X)
                class_values = data.domain.class_var.values
        if pred is not None:
            nan_values = False
            if np.any(np.isnan(data.Y)) or \
                    np.any(np.isnan(pred)):

                nan_values = True
                pred = data = None
            self.Error.invalid_values(shown=nan_values)
        if data is None:
            return

        return lda, lda._transformedData, pred, class_values

    def testdata_transform(self, data, projector):

        X = data
        Projector = projector
        transformed = Projector(X)
        return transformed

    def clear_outputs(self):

        self.Outputs.lda.send(None)

    def clear(self):
        """Reset the widget, clear controls"""
        self.tablemodel.clear()
        self.train_headers = []
        self._lda = None
        self.plot.clear_plot()
        self._CVE = None
        self.plotTwo.clear_plot()
        self._transformed = None
        self.train_pred = None
        self.train_class_values = None

    def _init_projector(self):
        self._lda_projector = LDA(solver="svd", shrinkage=None, priors=None,
                 n_components=MAX_COMPONENTS, store_covariance=False, tol=1e-4,
                 preprocessors=None)

#Scatter plot
    def _update_testdata_box(self):
        if self.test_data is None:
            self.testdata_box = False
        else:
            if self.testdata_box == False:
                self.testdata_classes_box = False
            self.test_datalabel = self.test_data.Y
            self.init_attr_values()
            self._setup_plot(self.attr_x, self.attr_y)

    def _update_testdata_classes_box(self):
        if self.test_data is None:
            self.testdata_box = False
            self.testdata_classes_box = False
        else:
            if self.testdata_classes_box == True:
                self.testdata_box = True
                self.test_datalabel = np.zeros(self.test_data.Y.shape, dtype=int)
                self.init_attr_values()
                self._setup_plot(self.attr_x, self.attr_y)
            else:
                self.test_datalabel = self.test_data.Y
                self.init_attr_values()
                self._setup_plot(self.attr_x, self.attr_y)

    def set_attr_from_combo(self):
        self.attr_changed()
        self.xy_changed_manually.emit(self.attr_x, self.attr_y)

    def attr_changed(self):
        self._setup_plot(self.attr_x, self.attr_y)
        self.commit()

    def _update_class_box(self):
        self.plot.clear_plot()
        self._setup_plot(self.attr_x, self.attr_y)

    def _update_legend_box(self):
        self.plot.clear_plot()
        self._setup_plot(self.attr_x, self.attr_y)

    def _setup_plot(self, x_axis, y_axis):
        self.plot.clear_plot()
        if self._lda is None:
            self.plot.clear_plot()
            return


        x=self._Transformed.X[:,self.domainIndexes[str(self.attr_x)]]
        y=self._Transformed.X[:,self.domainIndexes[str(self.attr_y)]]

        classes = self.train_classes.copy()
        if self.test_data is not None:
            if self.testdata_box is True:
                if self.testdata_classes_box ==True:
                    for kk in range(0,len(np.unique(self.testlabel))):
                        classes[len(self.train_classes)+kk] = 'Transformed testdata'

                    pass
                else:
                    for kk in range(0,len(np.unique(self._testdata_transformed.Y))):
                        classes[len(self.train_classes)+kk] = f'predicted {self.train_classes[kk]}'

        if self.class_box:
            self.PlotStyle = [
                dict(pen=None, symbolBrush=self.SYMBOLBRUSH[i], symbolPen=self.SYMBOLPEN[i], symbol='o', symbolSize=10,
                    name=classes[i]) for i in range(len(classes))]

            self.plot.update(x,y, Style=self.PlotStyle, labels=self.datalabel, x_axis_label=x_axis, y_axis_label=y_axis, legend=self.legend_box)
        else:
            self.Style = [
                dict(pen=None, symbolBrush=self.SYMBOLBRUSH[0], symbolPen=self.SYMBOLPEN[0], symbol='o', symbolSize=10,
                    name=classes[i]) for i in range(len(classes))]
            self.plot.update(x, y, Style=self.Style, labels=self.datalabel, x_axis_label=x_axis, y_axis_label=y_axis,legend=self.legend_box)

    def init_attr_values(self):
        if self.testdata_box:
            self.testlabel = testlabel = self.test_datalabel + np.max(self.train_datalabel) + 1
            self.datalabel = np.hstack((self.train_datalabel, testlabel))
            datatrans = np.vstack((self._transformed, self._testdata_transformed.X))
        else:
            self.datalabel = self.train_datalabel
            datatrans = self._transformed
        domain = numpy.array(['DF{}'.format(i + 1)
                              for i in range(datatrans.shape[1])],
                             dtype=object)

        for i in range(len(domain)):
            self.domainIndexes[domain[i]] = i

        proposed = [a for a in domain]
        dom = Domain(
            [ContinuousVariable(name, compute_value=lambda _: None)
             for name in proposed],
            metas=None)
        self._Transformed = Table(dom, datatrans, metas=None)
        self.xy_model.set_domain(dom)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x

#Confusion matrix
    def _param_changed(self):

        self._update_ConfMat()
        self._update_plot_two()

    def _update_ConfMat(self):
        def _isinvalid(x):
            return isnan(x) or isinf(x)


        if self.resampling == self.TestOnTrain:

            if self.train_pred is not None:
                cmatrix = confusion_matrix(self.train_data, self.train_pred)
                colsum = cmatrix.sum(axis=0)
                rowsum = cmatrix.sum(axis=1)
                n = len(cmatrix)
                diag = np.diag_indices(n)

                colors = cmatrix.astype(np.double)
                colors[diag] = 0
                if self.selected_quantity == 0:
                    normalized = cmatrix.astype(int)
                    formatstr = "{}"
                    div = np.array([colors.max()])
                else:
                    if self.selected_quantity == 1:
                        normalized = 100 * cmatrix / colsum
                        div = colors.max(axis=0)
                    else:
                        normalized = 100 * cmatrix / rowsum[:, np.newaxis]
                        div = colors.max(axis=1)[:, np.newaxis]
                    formatstr = "{:2.1f} %"
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
                        bkcolor = QColor.fromHsl(
                            [0, 240][i == j], 160,
                            255 if _isinvalid(col_val) else int(255 - 30 * col_val))
                        item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                        item.setData("trbl", BorderRole)
                        item.setToolTip("actual: {}\npredicted: {}".format(
                            self.train_headers[i], self.train_headers[j]))
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        self._set_item(i + 2, j + 2, item)

                bold_font = self.tablemodel.invisibleRootItem().font()
                bold_font.setBold(True)

                def _sum_item(value, border=""):
                    item = QStandardItem()
                    item.setData(value, Qt.DisplayRole)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled)
                    item.setFont(bold_font)
                    item.setData(border, BorderRole)
                    item.setData(QColor(192, 192, 192), BorderColorRole)
                    return item

                for i in range(n):
                    self._set_item(n + 2, i + 2, _sum_item(int(colsum[i]), "t"))
                    self._set_item(i + 2, n + 2, _sum_item(int(rowsum[i]), "l"))
                self._set_item(n + 2, n + 2, _sum_item(int(rowsum.sum())))
            else:
                return
        elif self.resampling == self.TestOnTest:
            if self.test_pred is not None:
                cmatrix = confusion_matrix(self.test_data, self.test_pred)
                colsum = cmatrix.sum(axis=0)
                rowsum = cmatrix.sum(axis=1)
                n = len(cmatrix)
                diag = np.diag_indices(n)

                colors = cmatrix.astype(np.double)
                colors[diag] = 0
                if self.selected_quantity == 0:
                    normalized = cmatrix.astype(int)
                    formatstr = "{}"
                    div = np.array([colors.max()])
                else:
                    if self.selected_quantity == 1:
                        normalized = 100 * cmatrix / colsum
                        div = colors.max(axis=0)
                    else:
                        normalized = 100 * cmatrix / rowsum[:, np.newaxis]
                        div = colors.max(axis=1)[:, np.newaxis]
                    formatstr = "{:2.1f} %"
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
                        bkcolor = QColor.fromHsl(
                            [0, 240][i == j], 160,
                            255 if _isinvalid(col_val) else int(255 - 30 * col_val))
                        item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                        item.setData("trbl", BorderRole)
                        item.setToolTip("actual: {}\npredicted: {}".format(
                            self.train_headers[i], self.train_headers[j]))
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        self._set_item(i + 2, j + 2, item)

                bold_font = self.tablemodel.invisibleRootItem().font()
                bold_font.setBold(True)

                def _sum_item(value, border=""):
                    item = QStandardItem()
                    item.setData(value, Qt.DisplayRole)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled)
                    item.setFont(bold_font)
                    item.setData(border, BorderRole)
                    item.setData(QColor(192, 192, 192), BorderColorRole)
                    return item

                for i in range(n):
                    self._set_item(n + 2, i + 2, _sum_item(int(colsum[i]), "t"))
                    self._set_item(i + 2, n + 2, _sum_item(int(rowsum[i]), "l"))
                self._set_item(n + 2, n + 2, _sum_item(int(rowsum.sum())))

        elif self.resampling == self.KFold:

            LDA = lda(solver="svd", shrinkage=None, priors=None,
            n_components=min(self.train_data.X.shape[1], len(self.classes)-1), store_covariance=False, tol=1e-4)
            kf = KFold(n_splits=self.NFolds[self.n_folds], shuffle=True, random_state=42)
            precmatrix = []
            for i in range(self.NFolds[self.n_folds]):
                precmatrix.append(numpy.zeros(shape=(len(self.train_data.domain.class_var.values),len(self.train_data.domain.class_var.values))))
            zaehler = 0


            for train_index, test_index in kf.split(self.train_data):

                train, test = self.train_data[train_index], self.train_data[test_index]
                LDA.fit(train.X, train.Y)
                Y_test_pred = LDA.predict(test.X)
                if zaehler == 0:
                    precmatrix[zaehler] = confusion_matrix(test, Y_test_pred)
                else:
                    precmatrix[zaehler] = precmatrix[zaehler-1] + confusion_matrix(test, Y_test_pred)
                zaehler = zaehler +1



            cmatrix = precmatrix[len(precmatrix)-1]
            colsum = cmatrix.sum(axis=0)
            rowsum = cmatrix.sum(axis=1)
            n = len(cmatrix)
            diag = np.diag_indices(n)

            colors = cmatrix.astype(np.double)
            colors[diag] = 0
            if self.selected_quantity == 0:
                normalized = cmatrix.astype(int)
                formatstr = "{}"
                div = np.array([colors.max()])
            else:
                if self.selected_quantity == 1:
                    normalized = 100 * cmatrix / colsum
                    div = colors.max(axis=0)
                else:
                    normalized = 100 * cmatrix / rowsum[:, np.newaxis]
                    div = colors.max(axis=1)[:, np.newaxis]
                formatstr = "{:2.1f} %"
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
                    bkcolor = QColor.fromHsl(
                        [0, 240][i == j], 160,
                        255 if _isinvalid(col_val) else int(255 - 30 * col_val))
                    item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                    item.setData("trbl", BorderRole)
                    item.setToolTip("actual: {}\npredicted: {}".format(
                        self.train_headers[i], self.train_headers[j]))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self._set_item(i + 2, j + 2, item)

            bold_font = self.tablemodel.invisibleRootItem().font()
            bold_font.setBold(True)

            def _sum_item(value, border=""):
                item = QStandardItem()
                item.setData(value, Qt.DisplayRole)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(Qt.ItemIsEnabled)
                item.setFont(bold_font)
                item.setData(border, BorderRole)
                item.setData(QColor(192, 192, 192), BorderColorRole)
                return item

            for i in range(n):
                self._set_item(n + 2, i + 2, _sum_item(int(colsum[i]), "t"))
                self._set_item(i + 2, n + 2, _sum_item(int(rowsum[i]), "l"))
            self._set_item(n + 2, n + 2, _sum_item(int(rowsum.sum())))
        elif self.resampling == self.StratifiedKFold:
            kf = StratifiedKFold(n_splits=self.sNFolds[self.sn_folds], shuffle=True, random_state=42)
            precmatrix = []
            LDA = lda(solver="svd", shrinkage=None, priors=None,
            n_components=min(self.train_data.X.shape[1], len(self.classes)-1), store_covariance=False, tol=1e-4)
            for i in range(self.sNFolds[self.sn_folds]):
                precmatrix.append(numpy.zeros(shape=(len(self.train_data.domain.class_var.values),len(self.train_data.domain.class_var.values))))
            zaehler = 0


            for train_index, test_index in kf.split(self.train_data, self.train_data.Y):

                train, test = self.train_data[train_index], self.train_data[test_index]
                LDA.fit(train.X, train.Y)
                Y_test_pred = LDA.predict(test.X)
                if zaehler == 0:
                    precmatrix[zaehler] = confusion_matrix(test, Y_test_pred)
                else:
                    precmatrix[zaehler] = precmatrix[zaehler-1] + confusion_matrix(test, Y_test_pred)
                zaehler = zaehler +1


            cmatrix = precmatrix[len(precmatrix)-1]
            colsum = cmatrix.sum(axis=0)
            rowsum = cmatrix.sum(axis=1)
            n = len(cmatrix)
            diag = np.diag_indices(n)

            colors = cmatrix.astype(np.double)
            colors[diag] = 0
            if self.selected_quantity == 0:
                normalized = cmatrix.astype(int)
                formatstr = "{}"
                div = np.array([colors.max()])
            else:
                if self.selected_quantity == 1:
                    normalized = 100 * cmatrix / colsum
                    div = colors.max(axis=0)
                else:
                    normalized = 100 * cmatrix / rowsum[:, np.newaxis]
                    div = colors.max(axis=1)[:, np.newaxis]
                formatstr = "{:2.1f} %"
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
                    bkcolor = QColor.fromHsl(
                        [0, 240][i == j], 160,
                        255 if _isinvalid(col_val) else int(255 - 30 * col_val))
                    item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                    item.setData("trbl", BorderRole)
                    item.setToolTip("actual: {}\npredicted: {}".format(
                        self.train_headers[i], self.train_headers[j]))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self._set_item(i + 2, j + 2, item)

            bold_font = self.tablemodel.invisibleRootItem().font()
            bold_font.setBold(True)

            def _sum_item(value, border=""):
                item = QStandardItem()
                item.setData(value, Qt.DisplayRole)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(Qt.ItemIsEnabled)
                item.setFont(bold_font)
                item.setData(border, BorderRole)
                item.setData(QColor(192, 192, 192), BorderColorRole)
                return item

            for i in range(n):
                self._set_item(n + 2, i + 2, _sum_item(int(colsum[i]), "t"))
                self._set_item(i + 2, n + 2, _sum_item(int(rowsum[i]), "l"))
            self._set_item(n + 2, n + 2, _sum_item(int(rowsum.sum())))
        elif self.resampling == self.LeaveOneOut:
            LDA = lda(solver="svd", shrinkage=None, priors=None,
            n_components=min(self.train_data.X.shape[1], len(self.classes)-1), store_covariance=False, tol=1e-4)
            kf = KFold(n_splits=self.train_data.Y.size, shuffle=True, random_state=42)
            precmatrix = []
            for i in range(self.train_data.Y.size):
                precmatrix.append(numpy.zeros(shape=(len(self.train_data.domain.class_var.values),len(self.train_data.domain.class_var.values))))
            zaehler = 0


            for train_index, test_index in kf.split(self.train_data):

                train, test = self.train_data[train_index], self.train_data[test_index]
                LDA.fit(train.X, train.Y)
                Y_test_pred = LDA.predict(test.X)
                if zaehler == 0:
                    precmatrix[zaehler] = confusion_matrix(test, Y_test_pred)
                else:
                    precmatrix[zaehler] = precmatrix[zaehler-1] + confusion_matrix(test, Y_test_pred)
                zaehler = zaehler +1


            cmatrix = precmatrix[len(precmatrix)-1]
            colsum = cmatrix.sum(axis=0)
            rowsum = cmatrix.sum(axis=1)
            n = len(cmatrix)
            diag = np.diag_indices(n)

            colors = cmatrix.astype(np.double)
            colors[diag] = 0
            if self.selected_quantity == 0:
                normalized = cmatrix.astype(int)
                formatstr = "{}"
                div = np.array([colors.max()])
            else:
                if self.selected_quantity == 1:
                    normalized = 100 * cmatrix / colsum
                    div = colors.max(axis=0)
                else:
                    normalized = 100 * cmatrix / rowsum[:, np.newaxis]
                    div = colors.max(axis=1)[:, np.newaxis]
                formatstr = "{:2.1f} %"
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
                    bkcolor = QColor.fromHsl(
                        [0, 240][i == j], 160,
                        255 if _isinvalid(col_val) else int(255 - 30 * col_val))
                    item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                    item.setData("trbl", BorderRole)
                    item.setToolTip("actual: {}\npredicted: {}".format(
                        self.train_headers[i], self.train_headers[j]))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self._set_item(i + 2, j + 2, item)

            bold_font = self.tablemodel.invisibleRootItem().font()
            bold_font.setBold(True)

            def _sum_item(value, border=""):
                item = QStandardItem()
                item.setData(value, Qt.DisplayRole)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(Qt.ItemIsEnabled)
                item.setFont(bold_font)
                item.setData(border, BorderRole)
                item.setData(QColor(192, 192, 192), BorderColorRole)
                return item

            for i in range(n):
                self._set_item(n + 2, i + 2, _sum_item(int(colsum[i]), "t"))
                self._set_item(i + 2, n + 2, _sum_item(int(rowsum[i]), "l"))
            self._set_item(n + 2, n + 2, _sum_item(int(rowsum.sum())))
            pass
        else:
            return

    def kfold_changed(self):
        self.resampling = OWPCALDA.KFold
        self._param_changed()

    def skfold_changed(self):
        self.resampling = OWPCALDA.StratifiedKFold
        self._param_changed()

    def _set_item(self, i, j, item):
        self.tablemodel.setItem(i, j, item)

    def _item(self, i, j):
        return self.tablemodel.item(i, j) or QStandardItem()

    def _init_table(self, nclasses):
        item = self._item(0, 2)
        item.setData("Predicted", Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.NoItemFlags)

        self._set_item(0, 2, item)
        item = self._item(2, 0)
        item.setData("Actual", Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        item.setFlags(Qt.NoItemFlags)
        self.tableview.setItemDelegateForColumn(0, gui.VerticalItemDelegate())
        self._set_item(2, 0, item)
        self.tableview.setSpan(0, 2, 1, nclasses)
        self.tableview.setSpan(2, 0, nclasses, 1)

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
            hor_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        else:
            hor_header.setDefaultSectionSize(110)
        self.tablemodel.setRowCount(nclasses + 3)
        self.tablemodel.setColumnCount(nclasses + 3)

#CVE plot
    def _on_cut_changed(self, components):
        if components == self.ncomponents \
                or self.ncomponents == 0 \
                or self._lda is not None \
                and components == len(self._CVE):
            return

        self.ncomponents = components
        if self._lda is not None:
            var = self._CVE[components - 1]
            if numpy.isfinite(var):
                self.CVE = int(var)

        self._invalidate_selection()

    def _nselected_components(self):
        """Return the number of selected components."""
        if self._CVE is None:
            return
        if self.ncomponents == 0:
            # Special "All" value
            max_comp = len(self._CVE)
        else:
            max_comp = self.ncomponents

        CVE_max = self._CVE[max_comp - 1]

        if CVE_max != numpy.floor(self.CVE):
            cut = max_comp
            assert numpy.isfinite(CVE_max)
            self.CVE = int(CVE_max)

        else:
            self.ncomponents = cut = numpy.searchsorted(
                self._CVE, self.CVE) + 1

        return cut

    def _setup_plot_Two(self):
        if self._CVE is None:
            return
        CVE = self._CVE
        cutpos = self._nselected_components()
        p = min(len(self._CVE), self.maxp)
        self.plotTwo.update(
            numpy.arange(1, p+1), [CVE[:p,]],
            [Qt.blue], cutpoint_x=cutpos, names=None)
        self._update_axis()

    def _update_axis(self):
        p = min(len(self._CVE), self.maxp)
        axis = self.plotTwo.getAxis("bottom")
        d = max((p-1)//(self.axis_labels-1), 1)
        axis.setTicks([[(i, str(i)) for i in range(1, p + 1, d)]])

    def _update_plot_two(self):
        self._cve()
        self._setup_plot_Two()

    def _cve(self):
        CVE = None
        self._CVE = None
        data = self.train_data[:]

        if self.resampling == self.TestOnTrain:
            if self.train_data is not None:
                training_data = self.train_data[:]

                def innerfunc(index):


                    training_data.X = data.X[:, :index+1]

                    LDA = lda(solver="svd", shrinkage=None, priors=None,
                    n_components=min(min(training_data.X.shape), len(numpy.unique(training_data.Y))-1), store_covariance=False, tol=1e-4)
                    LDA.fit(training_data.X, training_data.Y)
                    train_pred = LDA.predict(training_data.X)

                    if train_pred is not None:
                        cmatrix = confusion_matrix(training_data, train_pred)
                        summation = cmatrix.sum()
                        cve = (summation - numpy.trace(cmatrix))/summation
                        return cve


                CVElist = [innerfunc(index=i) for i in range(0, data.X.shape[1])]
                CVE = numpy.array(CVElist)
                self._CVE = CVE
        elif self.resampling == self.TestOnTest:

            training_data = self.train_data[:]
            if self.test_data is not None:
                test_data = self.test_data[:]
                testing_data = self.test_data[:]
                def innerfunc(index):
                    training_data.X = data.X[:, :index+1]
                    testing_data.X = test_data.X[:, :index+1]
                    LDA = lda(solver="svd", shrinkage=None, priors=None,
                    n_components=min(min(training_data.X.shape), len(numpy.unique(training_data.Y))-1), store_covariance=False, tol=1e-4)
                    LDA.fit(training_data.X, training_data.Y)
                    test_pred = LDA.predict(testing_data.X)

                    if test_pred is not None:
                        cmatrix = confusion_matrix(testing_data, test_pred)
                        summation = cmatrix.sum()
                        cve = (summation - numpy.trace(cmatrix))/summation
                        return cve

                CVElist = [innerfunc(index=i) for i in range(data.X.shape[1])]

                CVE = numpy.array(CVElist)
                self._CVE = CVE
        elif self.resampling == self.LeaveOneOut:
            if self.train_data is not None:
                pb = gui.ProgressBar(self, self.train_data.Y.size)
                kf = KFold(n_splits=self.train_data.Y.size, shuffle=True, random_state=42)
                CVElist = []
                cve_list_mat = []

                for train_index, test_index in kf.split(self.train_data):
                    train_data, test_data = self.train_data[train_index], self.train_data[test_index]
                    training_data = train_data[:]
                    testing_data = test_data[:]
                    def innerfunc(index):
                        training_data.X = train_data.X[:, :index+1]
                        testing_data.X = test_data.X[:, :index+1]
                        LDA = lda(solver="svd", shrinkage=None, priors=None,
                        n_components=min(min(training_data.X.shape), len(numpy.unique(training_data.Y))-1), store_covariance=False, tol=1e-4)
                        LDA.fit(training_data.X, training_data.Y)
                        test_pred = LDA.predict(testing_data.X)

                        if test_pred is not None:
                            cmatrix = confusion_matrix(testing_data, test_pred)
                            summation = cmatrix.sum()
                            #cve = (summation - numpy.trace(cmatrix))/summation
                            cve = (summation - numpy.trace(cmatrix))/summation
                            return cve
                    CVElist = [innerfunc(index=i) for i in range(data.X.shape[1])]
                    cve_list_mat.append(CVElist)
                    pb.advance()


                cve_mat =numpy.array(cve_list_mat)
                cve = numpy.sum(cve_mat,0)

                CVE = cve
                self._CVE = CVE
                pb.finish()
        elif self.resampling == self.KFold:
            if self.train_data is not None:
                pb = gui.ProgressBar(self, self.NFolds[self.n_folds])
                kf = KFold(n_splits=self.NFolds[self.n_folds], shuffle=True, random_state=42)
                CVElist = []
                cve_list_mat = []

                for train_index, test_index in kf.split(self.train_data):
                    train_data, test_data = self.train_data[train_index], self.train_data[test_index]
                    training_data = train_data[:]
                    testing_data = test_data[:]
                    def innerfunc(index):
                        training_data.X = train_data.X[:, :index+1]
                        testing_data.X = test_data.X[:, :index+1]
                        LDA = lda(solver="svd", shrinkage=None, priors=None,
                        n_components=min(min(training_data.X.shape), len(numpy.unique(training_data.Y))-1), store_covariance=False, tol=1e-4)
                        LDA.fit(training_data.X, training_data.Y)
                        test_pred = LDA.predict(testing_data.X)

                        if test_pred is not None:
                            cmatrix = confusion_matrix(testing_data, test_pred)
                            summation = cmatrix.sum()
                            #cve = (summation - numpy.trace(cmatrix))/summation
                            cve = (summation - numpy.trace(cmatrix))/summation
                            return cve
                    CVElist = [innerfunc(index=i) for i in range(data.X.shape[1])]
                    cve_list_mat.append(CVElist)
                    pb.advance()


                cve_mat =numpy.array(cve_list_mat)
                cve = numpy.sum(cve_mat,0)

                CVE = cve
                self._CVE = CVE
                pb.finish()
        elif self.resampling == self.StratifiedKFold:
            if self.train_data is not None:
                pb = gui.ProgressBar(self, self.NFolds[self.sn_folds])

                kf = StratifiedKFold(n_splits=self.sNFolds[self.sn_folds], shuffle=True, random_state=42)
                CVElist = []
                cve_list_mat = []

                for train_index, test_index in kf.split(self.train_data, self.train_data.Y):
                    train_data, test_data = self.train_data[train_index], self.train_data[test_index]
                    training_data = train_data[:]
                    testing_data = test_data[:]
                    def innerfunc(index):
                        training_data.X = train_data.X[:, :index + 1]
                        testing_data.X = test_data.X[:, :index + 1]
                        LDA = lda(solver="svd", shrinkage=None, priors=None,
                        n_components=min(min(training_data.X.shape), len(numpy.unique(training_data.Y))-1), store_covariance=False, tol=1e-4)
                        LDA.fit(training_data.X, training_data.Y)
                        test_pred = LDA.predict(testing_data.X)

                        if test_pred is not None:
                            cmatrix = confusion_matrix(testing_data, test_pred)
                            summation = cmatrix.sum()
                            #cve = (summation - numpy.trace(cmatrix))/summation
                            cve = (summation - numpy.trace(cmatrix))/summation
                            return cve
                    CVElist = [innerfunc(index=i) for i in range(data.X.shape[1])]
                    cve_list_mat.append(CVElist)
                    pb.advance()


                cve_mat =numpy.array(cve_list_mat)
                cve = numpy.sum(cve_mat,0)

                CVE = cve
                self._CVE = CVE
                pb.finish()
        return CVE

    def _update_selection_component_spin(self):

        if self._lda is None:
            self._invalidate_selection()
            return

        if self.ncomponents == 0 or self.ncomponents > len(self._CVE):

            cut = len(self._CVE)

        else:
            cut = self.ncomponents

        var = self._CVE[cut - 1]

        if numpy.isfinite(var):
            self.CVE = int(var)

        self.plotTwo.set_cut_point(cut)
        self._invalidate_selection()

    def _invalidate_selection(self):
        self.commit()


if __name__ == "__main__":  # pragma: no cover

    X = Table("iris")
    KF = KFold(n_splits=3, shuffle=True, random_state=42)
    KF.get_n_splits(X)

    train_index, test_index, bla = KF.split(X)
    X_train, X_test = X[test_index[0]], X[test_index[1]]

    WidgetPreview(OWPCALDA).run(set_train_data=X, set_test_data=X_test)