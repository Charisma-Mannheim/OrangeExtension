from xml.sax.saxutils import escape
import sys
import scipy.sparse as sp
import numbers
import numpy as np
from orangewidget.gui import tabWidget, createTabPage
import pyqtgraph as pg
from pyqtgraph.functions import mkPen
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from AnyQt.QtCore import Qt, QSize, QLineF, pyqtSignal as Signal
from AnyQt.QtGui import QPainter, QPen, QColor
from AnyQt.QtWidgets import QApplication, QGraphicsLineItem
from AnyQt.QtWidgets import QFormLayout
from Orange.data import Table, Domain, StringVariable, ContinuousVariable, DiscreteVariable, Variable
from Orange.data.util import get_unique_names
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.statistics.util import countnans, nanmean, nanmin, nanmax, nanstd
from Orange.preprocess import preprocess
from Orange.projection import PCA
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.slidergraph import SliderGraph
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output
from orangecontrib.extension.utils.Transform import Normalize, DoNothing, Center
from orangecontrib.extension.utils.LoggingDummyFile import PrinLog
from orangecontrib.extension.utils import ControlChart
from sklearn.metrics import mean_squared_error
from Orange.widgets.utils.itemmodels import DomainModel
from math import sqrt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import KFold
import time
from joblib import Parallel, delayed, parallel_backend
import copy
from Orange.widgets.settings import (
    Setting, ContextSetting, DomainContextHandler
)
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.plot import OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.visualize.owdistributions import LegendItem
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget

from orangecontrib.extension.utils import scattergraph


def ccw(a, b, c):
    """
    Checks whether three points are listed in a counterclockwise order.
    """
    ax, ay = (a[:, 0], a[:, 1]) if a.ndim == 2 else (a[0], a[1])
    bx, by = (b[:, 0], b[:, 1]) if b.ndim == 2 else (b[0], b[1])
    cx, cy = (c[:, 0], c[:, 1]) if c.ndim == 2 else (c[0], c[1])
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

def intersects(a, b, c, d):
    """
    Checks whether line segment a (given points a and b) intersects with line
    segment b (given points c and d).
    """
    return np.logical_and(ccw(a, c, d) != ccw(b, c, d),
                          ccw(a, b, c) != ccw(a, b, d))

def line_intersects_profiles(p1, p2, table):
    """
    Checks if a line intersects any line segments.

    Parameters
    ----------
    p1, p2 : ndarray
        Endpoints of the line, given x coordinate as p_[0]
        and y coordinate as p_[1].
    table : ndarray
        An array of shape m x n x p; where m is number of connected points
        for a individual profile (i. e. number of features), n is number
        of instances, p is number of coordinates (x and y).

    Returns
    -------
    result : ndarray
        Array of bools with shape of number of instances in the table.
    """
    res = np.zeros(len(table[0]), dtype=bool)
    for i in range(len(table) - 1):
        res = np.logical_or(res, intersects(p1, p2, table[i], table[i + 1]))
    return res

class LinePlotStyle:
    DEFAULT_COLOR = QColor(Qt.blue)

    UNSELECTED_LINE_ALPHA = 255

class LinePlotAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ticks = {}

    def set_ticks(self, ticks):
        if ticks:
            self._ticks = dict(enumerate(ticks, 1))
        else:
            return

    def tickStrings(self, values, scale, spacing):
        return [self._ticks.get(v * scale, "") for v in values]

class LinePlotViewBox(ViewBox):
    selection_changed = Signal(np.ndarray)

    def __init__(self):
        super().__init__(enableMenu=False)
        self._profile_items = None
        self._can_select = True
        self._graph_state = SELECT

        self.setMouseMode(self.PanMode)

    def set_graph_state(self, state):
        self._graph_state = state

    def get_selected(self, p1, p2):
        if self._profile_items is None:
            return np.array(False)
        return line_intersects_profiles(np.array([p1.x(), p1.y()]),
                                        np.array([p2.x(), p2.y()]),
                                        self._profile_items)
    def add_profiles(self, y):
        if sp.issparse(y):
            y = y.todense()
        self._profile_items = np.array(
            [np.vstack((np.full((1, y.shape[0]), i + 1), y[:, i].flatten())).T
             for i in range(y.shape[1])])

    def remove_profiles(self):
        self._profile_items = None

    def mouseClickEvent(self, event):
        if event.button() == Qt.RightButton:
            self.autoRange()
            self.enableAutoRange()
        else:
            event.accept()
            self.selection_changed.emit(np.array(False))

    def reset(self):
        self._profile_items = None
        self._can_select = True
        self._graph_state = SELECT

class LinePlotGraph(pg.PlotWidget):
    def __init__(self, parent, y_axis_label):
        self.bottom_axis = LinePlotAxisItem(orientation="bottom", maxTickLength=-5)
        super().__init__(parent, viewBox=LinePlotViewBox(),
                         background="w", enableMenu=False,
                         axisItems={"bottom": self.bottom_axis})
        self.left_axis = self.getAxis("left")
        self.left_axis.setLabel(y_axis_label)
        self.view_box = self.getViewBox()
        self.selection = set()
        self.getPlotItem().buttonsHidden = True
        self.setRenderHint(QPainter.Antialiasing, True)
        self.bottom_axis.labelText = "Features"
        self.bottom_axis.setLabel(axis=self.bottom_axis, text=self.bottom_axis.labelText or "")

    def select(self, indices):
        keys = QApplication.keyboardModifiers()
        indices = set(indices)
        if keys & Qt.ControlModifier:
            self.selection ^= indices
        elif keys & Qt.AltModifier:
            self.selection -= indices
        elif keys & Qt.ShiftModifier:
            self.selection |= indices
        else:
            self.selection = indices

    def reset(self):
        self.selection = set()
        self.view_box.reset()
        self.clear()
        self.getAxis('bottom').set_ticks(None)

class Profilespin_sel:
    def __init__(self, data, indices, color, graph):
        self.x_data = np.arange(1, data.X.shape[1] + 1)
        self.y_data = data.X
        self.indices = indices
        self.ids = data.ids
        self.color = color
        self.graph = graph
        self.graph_items = []
        self.__mean = nanmean(self.y_data, axis=0)
        self.__create_curves()

    def __create_curves(self):
        self.profiles = self._get_profiles_curve()
        self.graph_items = [
            self.profiles,
        ]

    def _get_profiles_curve(self):


        x, y, con = self.__get_disconnected_curve_data(self.y_data)
        color = QColor(self.color)
        color.setAlpha(LinePlotStyle.UNSELECTED_LINE_ALPHA)
        pen = self.make_pen(color)
        return pg.PlotCurveItem(x=x, y=y, connect=con, pen=pen, antialias=True)


    def remove_items(self):
        for item in self.graph_items:
            self.graph.removeItem(item)
        self.graph_items = []

    def set_visible_profiles(self, show_profiles=True, **_):
        if  show_profiles:

            self.graph.addItem(self.profiles)

        self.profiles.setVisible(show_profiles)


    def update_profiles_color(self, selection):
        color = QColor(self.color)
        alpha = LinePlotStyle.UNSELECTED_LINE_ALPHA
        color.setAlpha(alpha)
        x, y = self.profiles.getData()
        self.profiles.setData(x=x, y=y, pen=self.make_pen(color))


    @staticmethod
    def __get_disconnected_curve_data(y_data):
        m, n = y_data.shape
        x = np.arange(m * n) % n + 1
        y = y_data.A.flatten() if sp.issparse(y_data) else y_data.flatten()
        connect = np.ones_like(y, bool)
        connect[n - 1:: n] = False
        return x, y, connect

    @staticmethod
    def make_pen(color, width=1):
        pen = QPen(color, width)
        pen.setCosmetic(True)
        return pen

MAX_FEATURES = 10000
MAX_COMPONENTS = 100
LINE_NAMES = ["RMSECV by Eigen", "RMSECV row-wise"]
LINE_NAMES_TWO = ["component variance", "cumulative variance"]

class OWPCA(widget.OWWidget):

    name = "PCA"
    description = "Principal component analysis with a diagram of the root mean square error of reconstruction and prediction as well as \
    a scree plot of explained variance."
    icon = "icons/PCA.svg"
    priority = 1
    keywords = ["principal component analysis", "linear transformation"]

    class Inputs:

        data = Input("Data", Table)
        testdata = Input("Test Data", Table)

    class Outputs:
        data = Output("Data", Table, default=True)
        transformed_data = Output("Scores", Table, replaces=["Scores"])
        transformed_testdata = Output("Scores test data", Table)
        components = Output("Loadings", Table)
        outlier = Output("Outlier", Table)
        inlier = Output("Outlier corrected Dataset", Table)
        pca = Output("PCA", PCA, dynamic=False)

    Clvl = [68.3, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 95.4, 99.7, 99.9]

    variance_covered = settings.Setting(100)
    RMSECV = settings.Setting(10)
    auto_commit = settings.Setting(True)
    standardize = settings.Setting(False)
    centering = settings.Setting(True)
    maxp = settings.Setting(30)
    axis_labels = settings.Setting(10)
    ncomponents = settings.Setting(3)
    class_box = settings.Setting(True)
    legend_box = settings.Setting(False)
    confidence = settings.Setting(6)
    Principal_Component = Setting(1)
    show_profiles = Setting(True)
    class_box = Setting(True)
    legend_box = Setting(False)
    testdata_box = Setting(False)
    testdata_classes_box = Setting(False)
    selection = ContextSetting(set())
    score_x = ContextSetting(None)
    score_y = ContextSetting(None)

    graph_name = "plot.plotItem"

    class Warning(widget.OWWidget.Warning):
        trivial_components = widget.Msg(
            "All components of the PCA are trivial (explain 0 variance). "
            "Input data is constant (or near constant).")
        no_display_option = Msg("No display option is selected.")

    class Error(widget.OWWidget.Error):
        no_features = widget.Msg("At least 1 feature is required")
        no_instances = widget.Msg("At least 1 data instance is required")
        no_traindata = widget.Msg("No train data submitted")
        not_enough_attrs = Msg("Need at least one continuous feature.")
        no_valid_loadings = Msg("No plot due to no valid data.")
        loading_not_available = Msg("Requested loadings not available.")
        
    class Information(OWWidget.Information):
        hidden_instances = Msg("Instances with unknown values are not shown.")
        too_many_features = Msg("Data has too many features. Only first {}"
                                " are shown.".format(MAX_FEATURES))

    xy_changed_manually = Signal(Variable, Variable)
    common_options = dict(
        labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True,
        contentsLength=14
    )

    def __init__(self):

        dmod = DomainModel
        self.xy_model = DomainModel(dmod.MIXED, valid_types=ContinuousVariable)
        super().__init__()
        self.parallel = Parallel(n_jobs=-1, pre_dispatch='2*n_jobs', prefer="threads")
        self.data = None
        self.testdata = None
        self._testdata_transformed = None
        self._pca = None
        self._transformed = None
        self._variance_ratio = None
        self._cumulative = None
        self.domainIndexes = {}
        self.domainIndexesScore = {}
        self._RMSECV = None
        self._rmseCV = None
        self._statistics = None
        self.train_classes = None
        self.datalabel = None
        self.datalabelqt = None
        self.classes = None
        self.outlier_metas = None
        self.inlier_data = None
        self.outlier_data = None
        self.components = None
        self.graph_variables = []
        self.__spin_selection = []
        self.loadings = None
        self.testlabel = None

        self.SYMBOLBRUSH = [(0, 204, 204, 180), (51, 255, 51, 180), (255, 51, 51, 180), (0, 128, 0, 180),  \
                       (195, 46, 212, 180), (250, 194, 5, 180), (55, 55, 55, 180), (0, 114, 189, 180), (217, 83, 25, 180), (237, 177, 32, 180), \
                       (126, 47, 142, 180), (119, 172, 180)]

        self.SYMBOLPEN = [(0, 204, 204, 255), (51, 255, 51, 255), (255, 51, 51, 255), (0, 128, 0, 255),  \
                       (195, 46, 212, 255), (250, 194, 5, 255), (55, 55, 55, 255), (0, 114, 189, 255), (217, 83, 25, 255), (237, 177, 32, 255), \
                       (126, 47, 142, 255), (119, 172, 255)]

        self._init_projector()

        box = gui.vBox(self.controlArea, "Components Selection")
        form = QFormLayout()
        box.layout().addLayout(form)

        self.components_spin = gui.spin(
            box, self, "ncomponents", 1, MAX_COMPONENTS,
            callback=self._update_selection_components_spin,
            keyboardTracking=False, addToLayout=False
        )
        self.components_spin.setSpecialValueText("All")

        self.variance_spin = gui.spin(
            box, self, "variance_covered", 1, 100,
            callback=self._update_selection_variance_spin,
            keyboardTracking=False
        )
        self.variance_spin.setSuffix("%")

        form.addRow("Components:", self.components_spin)
        form.addRow("Explained variance:", self.variance_spin)

        # Options
        self.options_box = gui.vBox(self.controlArea, "Options")
        form = QFormLayout()
        box.layout().addLayout(form)

        self.standardize_box = gui.checkBox(
            self.options_box, self, "standardize", "standardize variables", callback=self._update_standardize
        )

        self.center_box = gui.checkBox(
            self.options_box, self, "centering", "mean-center variables", callback=self._update_centering
        )

        self.maxp_spin = gui.spin(
            self.options_box, self, "maxp", 1, MAX_COMPONENTS,
            label="Show only first", callback=self._setup_plot
        )

        class_box = gui.vBox(self.controlArea, "Control chart and score plot options")

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

        gui.comboBox(
            class_box, self, "confidence", label="Shown level of confidence: ",
            items=[str(x) for x in self.Clvl],
            orientation=Qt.Horizontal, callback=self._param_changed)

        lbox = gui.vBox(self.controlArea, "Display Loadings for")
        lform = QFormLayout()
        lbox.layout().addLayout(lform)


        self.component_spin = gui.spin(
            lbox, self, "Principal_Component", 1, MAX_FEATURES,
            callback=self._update_selection_component_spin,
            keyboardTracking=False
        )
        lform.addRow("Component:", self.component_spin)

        sbox = gui.vBox(self.controlArea, "Discriminant function selection")
        sform = QFormLayout()
        sbox.layout().addLayout(sform)

        dmod = DomainModel
        self.xy_modelScore = DomainModel(dmod.MIXED, valid_types=ContinuousVariable)
        self.cb_score_x = gui.comboBox(
            sbox, self, "score_x", label=None,
            callback=self.set_attr_from_combo,
            model=self.xy_modelScore, **self.common_options,
            searchable=True)
        self.cb_score_y = gui.comboBox(
            sbox, self, "score_y", label=None,
            callback=self.set_attr_from_combo,
            model=self.xy_modelScore, **self.common_options,
            searchable=True)

        sform.addRow("Axis x:", self.cb_score_x)
        sform.addRow("Axis y:", self.cb_score_y)

        self.controlArea.layout().addStretch()
        gui.auto_apply(self.controlArea, self, "auto_commit")

        self.plot = SliderGraph(
            "Principal Components", "RMSECV",
            self._on_cut_changed)
        self.plotTwo = SliderGraph(
            "Principal Components", "Proportion of variance",
            self._on_cut_changed_two)

        self.plotThree = ControlChart.ScatterGraph(callback=None)
        self.loadingsplot = LinePlotGraph(self, y_axis_label="Loadings")
        self.scoreplot = scattergraph.ScatterGraph(callback=None)


        tabs = tabWidget(self.mainArea)

        # graph tab
        tab = createTabPage(tabs, "Error")
        tab.layout().addWidget(self.plot)

        # graph 2 tab
        tab = createTabPage(tabs, "Scree")
        tab.layout().addWidget(self.plotTwo)

        tab = createTabPage(tabs, "Q residuals vs. Hotellings T²")
        tab.layout().addWidget(self.plotThree)

        tab = createTabPage(tabs, "Loadings plot")
        tab.layout().addWidget(self.loadingsplot)

        tab = createTabPage(tabs, "Score plot")
        tab.layout().addWidget(self.scoreplot)


        self._update_centering()
##Process input data
    @Inputs.data
    def set_data(self, data):

        self.clear_messages()
        self.clear()
        self.information()
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

        if data is not None:

            self._init_projector()
            self.data = data

            if hasattr(self.data.domain.class_var, 'values'):
                self.classes = np.arange(0, len(self.data.domain.class_var.values))
                self.train_classes = {int(self.classes[i]): self.data.domain.class_var.values[i] for i in
                                      np.arange(0, len(self.data.domain.class_var.values))}
            else:
                self.classes = None

            self.fit()
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)
            self.init_score_values()
            self._setup_score_plot(self.score_x, self.score_y)
            self.unconditional_commit()

    @Inputs.testdata
    def set_testdata(self, data):
        self.testdata = None
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
        if data is not None:
            self.testdata = data
            if self.data is None:
                self.Error.no_traindata()
                return
            self._testdata_transformed = self.testdata_transform(self.testdata, self._pca)

        if self.testdata_box:
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)
            self.init_score_values()
            self._setup_score_plot(self.score_x, self.score_y)
        self.unconditional_commit()

    def testdata_transform(self, data, projector):

        X = data
        Projector = projector
        transformed = Projector(X)
        return transformed

    def fit(self):
        self.clear()
        self.Warning.trivial_components.clear()
        if self.data is None:
            return
        data = self.data
        if self.standardize:
            self._pca_projector.preprocessors = \
                self._pca_preprocessors + [preprocess.Normalize(center=True)]
        else:
            self._pca_projector.preprocessors = self._pca_preprocessors
        if not isinstance(data, SqlTable):
            Data = data[:]
            pca = self._pca_projector(data)
            variance_ratio = pca.explained_variance_ratio_
            cumulative = np.cumsum(variance_ratio)
            if len(pca.components_) >= 30:
                COMPONENTS = 30
            else:
                COMPONENTS = pca.components_.shape[1]
            pb = gui.ProgressBar(self, 10)
            pbtwo = gui.ProgressBar(self, 10)
            rmseCV, RMSECV = self.rmseCV(Data, COMPONENTS, pb=[pb,pbtwo])
            pb.finish()
            pbtwo.finish()
            if np.isfinite(cumulative[-1]):
                self.components_spin.setRange(0, len(cumulative))
                self._pca = pca
                self._variance_ratio = variance_ratio
                self._cumulative = cumulative
                self._RMSECV = RMSECV
                self._rmseCV = rmseCV
                self._COMPONENTS = COMPONENTS
                self._setup_plot()

            else:
                self.Warning.trivial_components()
            self.unconditional_commit()
            if self._pca is not None:
                self.loadings = self.components
                self.check_loadings()
                self.setup_loadingsplot()

    def preprocess(data, preprocessors):

        for pp in preprocessors:
            data = pp(data)
        return data

    def rmseCV(self, data, ncomponents, pb):
        normalizer = Normalize()
        centerer = Center()
        if self.standardize == True and self.centering == False:
            X_preprocessed = normalizer.fit_transform(data.X)
            XArray = np.copy(X_preprocessed)
        elif self.centering == True and self.standardize == False:
            X_preprocessed = centerer.fit_transform(data.X)
            XArray = np.copy(X_preprocessed)

        n_splits = 10
        Kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmseCV_list_matrix = []
        pca = sklearnPCA(random_state=42, svd_solver='full')
        for train_index, test_index in Kf.split(XArray):
            X_train, X_test = XArray[train_index], XArray[test_index]
            pca.fit(X_train)
            def innerfunctwo(index):
                X_testtransformed = np.dot(X_test, pca.components_.T[:,:index])
                X_test_pred = np.dot(X_testtransformed, pca.components_[:index,:])
                if self.standardize == True and self.centering == False:
                    X_testIT = normalizer.inverse_transform(X_test)
                    X_test_predIT = normalizer.inverse_transform(X_test_pred)
                elif self.centering == True and self.standardize == False:
                    X_testIT = centerer.inverse_transform(X_test)
                    X_test_predIT = centerer.inverse_transform(X_test_pred)
                rmseCV = np.sum((X_testIT - X_test_predIT)**2)
                return rmseCV

            rmseCV_list = [innerfunctwo(index=index) for index in range(1,ncomponents+1,1)]
            rmseCVarray = np.array(rmseCV_list)
            rmseCV_list_matrix.append(rmseCVarray)
            pb[1].advance()
        rmseCVMatrix = np.array(rmseCV_list_matrix)
        rmseCV = np.sum(rmseCVMatrix,0)
        rmseCV = np.sqrt(rmseCV/(XArray.shape[0]*XArray.shape[1]))
        rmseCV = rmseCV/rmseCV.max()
        rmseCVrowwise = rmseCV.reshape((len(rmseCV), 1))


        KF = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        i = 0
        pcatwo = sklearnPCA(random_state=42, svd_solver='full')
        for train_index, test_index in KF.split(XArray):
            X_train, X_test = XArray[train_index], XArray[test_index]
            pcatwo.fit(X_train)
            X_test_transposed = X_test.T
            if X_test_transposed.shape[0] >= 5:
                numberOfFolds = 5
            else:
                numberOfFolds = X_test_transposed.shape[0]
            def innerfunctwo(index):
                P = pcatwo.components_.T[:, :index]
                kf = KFold(n_splits=numberOfFolds, shuffle=True, random_state=42)
                residual_list = []
                for reduced_index, rest_index in kf.split(X_test_transposed):
                    X_test_reduced_transposed, X_test_rest_transposed = X_test_transposed[reduced_index], X_test_transposed[rest_index]
                    X_test_reduced = X_test_reduced_transposed.T
                    P_reduced = P[reduced_index]
                    P_rest = P[rest_index]
                    X_test_rest = X_test_rest_transposed.T
                    t_reduced = np.dot(X_test_reduced, P_reduced)
                    X_test_rest_pred = np.dot(t_reduced, P_rest.T)
                    if self.standardize == True and self.centering == False:
                        X_test_restIT = normalizer.inverse_transform(X_test_rest, rest_index)
                        X_test_rest_predIT = normalizer.inverse_transform(X_test_rest_pred, rest_index)
                    elif self.centering == True and self.standardize == False:
                        X_test_restIT = centerer.inverse_transform(X_test_rest, rest_index)
                        X_test_rest_predIT = centerer.inverse_transform(X_test_rest_pred, rest_index)
                    residual_list.append(X_test_restIT-X_test_rest_predIT)
                c = 0
                for cols in residual_list:
                    if c == 0:
                        residuals = cols
                        c = c + 1
                    else:
                        residuals = np.hstack((residuals, cols))
                return residuals
            ResidualsPerComp_list = [innerfunctwo(index=index) for index in range(1,ncomponents+1,1)]
            k = 0
            if i == 0:
                RMSECV = np.empty((ncomponents, n_splits))
                for mat in ResidualsPerComp_list:
                    RMSECV[k,i] = np.sum(mat ** 2)
                    k = k + 1
                i = i + 1
            else:
                for mat in ResidualsPerComp_list:
                    RMSECV[k, i] = np.sum(mat ** 2)
                    k = k + 1
                i = i + 1
            pb[0].advance()
        RMSECV = np.sum(RMSECV, axis=1)
        RMSECV = np.sqrt(RMSECV/(XArray.shape[0]*XArray.shape[1]))
        RMSECV = RMSECV/RMSECV.max()
        rmseCVbyEigen = RMSECV.reshape((len(RMSECV),1))
        return rmseCVrowwise, rmseCVbyEigen

    def setProgressValue(self, value):

        self.progressBarSet(value)

    def _init_projector(self):

        self._pca_projector = PCA(n_components=MAX_COMPONENTS, random_state=0)
        self._pca_projector.component = self.ncomponents
        self._pca_preprocessors = PCA.preprocessors

##Widget properties

    def commit(self):
        inlier = outlier  = transformed = transformed_testdata = data = components = None
        if self._pca is not None:
            if self._transformed is None:
                self._transformed = self._pca(self.data)

            transformed = self._transformed
            explVar = np.array(self._variance_ratio, ndmin=2)
            explVarFlo = [k for k in explVar]
            a = [str(i)  for i in transformed.domain.attributes[:self.ncomponents]]
            b = [" (" + str(np.around(i*100, decimals=2)) + "%) " for i in explVarFlo[0][:self.ncomponents]]
            Domainfinal = [a[i] + b[i] for i in range(self.ncomponents)]
            domainFinal = Domain(
                [ContinuousVariable(name, compute_value=lambda _: None)
                              for name in Domainfinal],
                self.data.domain.class_vars,
                self.data.domain.metas
            )
            domain = Domain(
                transformed.domain.attributes[:self.ncomponents],
                self.data.domain.class_vars,
                self.data.domain.metas
            )
            transformed = transformed.from_table(domain, transformed)
            transformed.domain = domainFinal
            if self._testdata_transformed is not None:
                transformed_testdata = self._testdata_transformed
                domainzwo = Domain(
                    transformed_testdata.domain.attributes[:self.ncomponents],
                    self.data.domain.class_vars,
                    self.data.domain.metas
                )
                transformed_testdata = transformed_testdata.from_table(domainzwo, transformed_testdata)
                transformed_testdata.domain = domainFinal
            else:
                transformed_testdata = None
            proposed = [a.name for a in self._pca.orig_domain.attributes]
            meta_name = get_unique_names(proposed, 'components')
            dom = Domain(
                [ContinuousVariable(name, compute_value=lambda _: None)
                 for name in proposed],
                metas=[StringVariable(name=meta_name)])
            metas = np.array([['PC{}'.format(i + 1)
                                  for i in range(self.ncomponents)]],
                                dtype=object).T

            metas4 = np.array([['id'
                                   ]],
                                 dtype=object)

            components = Table(dom, self._pca.components_[:self.ncomponents,:],
                               metas=metas)
            components.name = 'components'

            data_dom = Domain(
                self.data.domain.attributes,
                self.data.domain.class_vars,
                self.data.domain.metas + domain.attributes)
            data = Table.from_numpy(
                data_dom, self.data.X, self.data.Y,
                np.hstack((self.data.metas, transformed.X)),
                ids=self.data.ids)

            inlier = self.inlier_data
            outlier = self.outlier_data
        self._pca_projector.component = self.ncomponents
        self.Outputs.data.send(data)
        self.Outputs.transformed_data.send(transformed)
        self.Outputs.transformed_testdata.send(transformed_testdata)
        self.Outputs.components.send(components)
        self.Outputs.pca.send(self._pca_projector)
        self.Outputs.outlier.send(outlier)
        self.Outputs.inlier.send(inlier)
        self.components = components

    def clear(self):

        self._pca = None
        self._transformed = None
        self._Transformed = None
        self._variance_ratio = None
        self._cumulative = None
        self.plot.clear_plot()
        self.plotTwo.clear_plot()
        self.plotThree.clear_plot()
        self._RMSECV = None
        self._rmseCV = None

    def clear_outputs(self):
        self.Outputs.data.send(None)
        self.Outputs.transformed_data.send(None)
        self.Outputs.transformed_testdata.send(None)
        self.Outputs.components.send(None)
        self.Outputs.pca.send(self._pca_projector)
        self.Outputs.outlier.send(None)
        self.Outputs.inlier.send(None)

    def _invalidate_selection(self):
        self.commit()

    def send_report(self):

        if self.data is None:
            return
        self.report_plot("Reconstruction Error", self.plot)
        self.report_plot("Explained Variance", self.plotTwo)
        self.report_plot("T²/Q", self.plotThree)
        self.report_plot(f'Loadings plot of principal component {self.Principal_Component}', self.loadingsplot)
        self.report_plot(f'Score plot {self.score_x} vs. {self.score_y}', self.scoreplot)

    @classmethod
    def migrate_settings(cls, settings, version):
        if "variance_covered" in settings:
            # Due to the error in gh-1896 the variance_covered was persisted
            # as a NaN value, causing a TypeError in the widgets `__init__`.
            vc = settings["variance_covered"]
            if isinstance(vc, numbers.Real):
                if np.isfinite(vc):
                    vc = int(vc)
                else:
                    vc = 100
                settings["variance_covered"] = vc
        if settings.get("ncomponents", 0) > MAX_COMPONENTS:
            settings["ncomponents"] = MAX_COMPONENTS

        # Remove old `decomposition_idx` when SVD was still included
        settings.pop("decomposition_idx", None)

        # Remove RemotePCA settings
        settings.pop("batch_size", None)
        settings.pop("address", None)
        settings.pop("auto_update", None)
##Plot one and two
    def _setup_plot(self):
        if self._pca is None:
            self.plot.clear_plot()
            self.plot_two.clear_plot()
            return

        RMSECV = self._RMSECV
        rmseCV = self._rmseCV
        cutpos = self._nselected_components()
        p = min(len(self._RMSECV), self.maxp)

        self.plot.update(
            np.arange(1, p+1), [RMSECV[:p,0], rmseCV[:p,0]],
            [Qt.blue, Qt.red], cutpoint_x=cutpos, names=LINE_NAMES)
        self.plot.setRange(yRange=(min(yi.min() for yi in [rmseCV, RMSECV]), max(yi.max() for yi in [rmseCV, RMSECV])))
        explained_ratio = self._variance_ratio
        explained = self._cumulative
        cutposTwo = self._nselected_components_two()
        pTwo = min(len(self._variance_ratio), self.maxp)

        self.plotTwo.update(
            np.arange(1, p+1), [explained_ratio[:pTwo], explained[:pTwo]],
            [Qt.red, Qt.darkYellow], cutpoint_x=cutposTwo, names=LINE_NAMES_TWO)

        self._update_axis()

    def _on_cut_changed(self, components):

        if components == self.ncomponents:
            return
        self._on_cut_changed_two(components)
        self.ncomponents = components
        if self._pca is not None:
            self.loadings = self.components
            self.check_loadings()
            self.setup_loadingsplot()

    def _on_cut_changed_two(self, components):

        if components == self.ncomponents:
            return
        self.ncomponents = components
        if self._pca is not None:
            var = self._cumulative[components - 1]
            if np.isfinite(var):
                self.variance_covered = int(var * 100)
        self._update_selection_components_spin()
        self._update_selection_variance_spin()
        self._invalidate_selection()
        if self._pca is not None:
            self.loadings = self.components
            self.check_loadings()
            self.setup_loadingsplot()

    def _update_selection_components_spin(self):

        if self._pca is None:
            self._invalidate_selection()
            return

        if self.ncomponents == 0:

            cut = self._COMPONENTS

        else:
            cut = self.ncomponents
        var = self._cumulative[cut - 1]
        if np.isfinite(var):
            self.variance_covered = int(var * 100)
        self.ncomponents = cut
        self.plot.set_cut_point(cut)
        self.plotTwo.set_cut_point(cut)
        self.init_attr_values()
        self._setup_plotThree(self.attr_x, self.attr_y)
        self.init_score_values()
        self._setup_score_plot(self.score_x, self.score_y)
        if self.ncomponents < self.Principal_Component:
            self.Principal_Component = self.ncomponents
        self._invalidate_selection()
        if self._pca is not None:
            self.loadings = self.components
            self.check_loadings()
            self.setup_loadingsplot()

    def _update_selection_variance_spin(self):

        if self._pca is None:
            self._invalidate_selection()
            return
        cut = np.searchsorted(self._cumulative,
                                 self.variance_covered / 100.0) + 1
        cut = min(cut, len(self._cumulative))
        self.ncomponents = cut
        if self.ncomponents < self.Principal_Component:
            self.Principal_Component = self.ncomponents
        self.plot.set_cut_point(cut)
        self.plotTwo.set_cut_point(cut)
        self.init_attr_values()
        self._setup_plotThree(self.attr_x, self.attr_y)
        self.init_score_values()
        self._setup_score_plot(self.score_x, self.score_y)
        self._invalidate_selection()
        if self._pca is not None:
            self.loadings = self.components
            self.check_loadings()
            self.setup_loadingsplot()

    def _nselected_components(self):

        """Return the number of selected components."""
        if self._pca is None:
            return 0
        if self.ncomponents == 0:
            max_comp = self._COMPONENTS
        else:
            max_comp = self.ncomponents
        RMSE_max = self._RMSECV[max_comp - 1]
        cut = max_comp
        assert np.isfinite(RMSE_max)
        return cut

    def _nselected_components_two(self):
        """Return the number of selected components."""
        if self._pca is None:
            return 0
        if self.ncomponents == 0:
            max_comp = len(self._variance_ratio)
        else:
            max_comp = self.ncomponents
        var_max = self._cumulative[max_comp - 1]
        if var_max != np.floor(self.variance_covered / 100.0):
            cut = max_comp
            assert np.isfinite(var_max)
            self.variance_covered = int(var_max * 100)
        else:
            self.ncomponents = cut = np.searchsorted(
                self._cumulative, self.variance_covered / 100.0) + 1
        return cut

    def _update_axis(self):
        p = min(len(self._RMSECV), self.maxp)
        axis = self.plot.getAxis("bottom")
        d = max((p-1)//(self.axis_labels-1), 1)
        axis.setTicks([[(i, str(i)) for i in range(1, p + 1, d)]])

    def setup_plot(self):
        super().setup_plot()

##T²/Q plot
    def init_attr_values(self):

        normalizer = Normalize()
        centerer = Center()
        if self.testdata_box:
            if self.testdata_classes_box == True:
                testlabelqt = np.full(self.testdata.Y.shape, np.max(self.data.Y) + 1, dtype=int)
            else:
                testlabelqt = self._testdata_transformed.Y + np.max(self.data.Y)+1
            self.datalabelqt = np.hstack((self.data.Y, testlabelqt))
            data = np.vstack((self.data.X, self.testdata.X))
        else:
            self.datalabelqt = self.data.Y
            data = self.data.X

        if self.standardize == True and self.centering == False:
            X_preprocessed = normalizer.fit_transform(data)
            XArray = np.copy(X_preprocessed)
        elif self.centering == True and self.standardize == False:
            X_preprocessed = centerer.fit_transform(data)
            XArray = np.copy(X_preprocessed)

        else:
            return

        pca = sklearnPCA(n_components=self.ncomponents, random_state=42, svd_solver='full')
        pca.fit(XArray)
        T = Xtransformed = pca.transform(XArray)
        P = pca.components_[:self.ncomponents, :]
        X_pred = np.dot(T, P) + pca.mean_

        if self.standardize == True and self.centering == False:
            X_predIT = normalizer.inverse_transform(X_pred)
        elif self.centering == True and self.standardize == False:
            X_predIT = centerer.inverse_transform(X_pred)

        Err = data-X_predIT
        #Err = XArray - X_pred
        Q = np.sum(Err ** 2, axis=1)
        Q = Q.reshape((len(Q),1))

        # Calculate Hotelling's T-squared (note that data are normalised by default)
        Tsq = np.sum((T / np.std(T, axis=0)) ** 2, axis=1)
        Tsq = Tsq.reshape((len(Tsq),1))

        statistics = np.hstack((Tsq, Q))


        domain = np.array(['T²', 'Q'],
                            dtype=object)

        for i in range(len(domain)):
            self.domainIndexes[domain[i]] = i

        proposed = [a for a in domain]

        dom = Domain(
            [ContinuousVariable(name, compute_value=lambda _: None)
             for name in proposed],
            metas=None)
        self._statistics = Table(dom, statistics, metas=None)
        self.xy_model.set_domain(dom)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x

    def _param_changed(self):

        self.plotThree.clear_plot()
        self.init_attr_values()
        self._setup_plotThree(self.attr_x, self.attr_y)

    def _setup_plotThree(self, x_axis, y_axis):

        self.plotThree.clear_plot()
        if self.data is None:
            self.plotThree.clear_plot()
            return


        x=self._statistics.X[:,self.domainIndexes[str(self.attr_x)]]
        y=self._statistics.X[:,self.domainIndexes[str(self.attr_y)]]
        y = y/y.max()
        #x = x/x.max()
        classes = self.train_classes.copy()
        if self.testdata is not None:
            if self.testdata_box is True:
                if self.testdata_classes_box ==True:
                    for kk in range(0,len(np.unique(self.testlabel))):
                        classes[len(self.train_classes)+kk] = 'Transformed testdata'

                    pass
                else:
                    for kk in range(0,len(np.unique(self._testdata_transformed.Y))):
                        classes[len(self.train_classes)+kk] = f'predicted {self.train_classes[kk]}'

        # set the confidence level
        conf = self.Clvl[self.confidence]/100
        from scipy.stats import f
        # Calculate confidence level for T-squared from the ppf of the F distribution
        Tsq_conf = f.ppf(q=conf, dfn=self.ncomponents,dfd=self.data.X.shape[0]) * self.ncomponents * (self.data.X.shape[0] - 1) / (self.data.X.shape[0] - self.ncomponents)
        # Estimate the confidence level for the Q-residuals
        Qsorted = np.sort(y)
        i = len(Qsorted)-1

        while 1 - np.sum(y > Qsorted[i]) / np.sum(y > 0) > conf:
            i -= 1
            if i == 0:
                break
        Q_conf = Qsorted[i]
        data = self.data[:]
        if self.testdata:
            if self.testdata_box:
                data.X = np.vstack((self.data.X, self.testdata.X))
                data.W = np.vstack((self.data.W, self.testdata.W))
                data.Y = np.hstack((self.data.Y, self.testdata.Y))
                data.ids = np.hstack((self.data.ids, self.testdata.ids))
                data.metas = np.vstack((self.data.metas, self.testdata.metas))
            #else:
                #data = self.data[:]
        #else:
            #data = self.data[:]




        outlier_index_Q = y > Q_conf
        outlier_index_T = x > Tsq_conf
        outlier_index = outlier_index_Q + outlier_index_T
        inlier_index = ~outlier_index

        inlier_data = copy.copy(data)
        inlier_data.X = data.X[inlier_index, :]
        inlier_data.W = data.W[inlier_index, :]
        inlier_data.Y = data.Y[inlier_index]
        inlier_data.ids = data.ids[inlier_index]
        inlier_data.metas = data.metas[inlier_index]
        self.inlier_data = inlier_data
        outlier_data = copy.copy(data)
        outlier_data.X = data.X[outlier_index, :]
        outlier_data.W = data.W[outlier_index, :]
        outlier_data.Y = data.Y[outlier_index]
        outlier_data.ids = data.ids[outlier_index]
        outlier_data.metas = data.metas[outlier_index]
        self.outlier_data = outlier_data


        if self.classes is not None:

            if self.class_box:

                self.PlotStyle = [
                    dict(pen=None, symbolBrush=self.SYMBOLBRUSH[i], symbolPen=self.SYMBOLPEN[i], symbol='o', symbolSize=10,
                        name=classes[i]) for i in range(len(classes))]

                self.plotThree.update(x,y, Style=self.PlotStyle, labels=self.datalabelqt, x_axis_label=x_axis, y_axis_label=y_axis, legend=self.legend_box, tucl=Tsq_conf, qucl=Q_conf)
            else:

                self.Style = [
                    dict(pen=None, symbolBrush=self.SYMBOLBRUSH[0], symbolPen=self.SYMBOLPEN[0], symbol='o', symbolSize=10,
                        name=classes[i]) for i in range(len(classes))]
                self.plotThree.update(x, y, Style=self.Style, labels=self.datalabelqt, x_axis_label=x_axis, y_axis_label=y_axis,legend=self.legend_box, tucl=Tsq_conf, qucl=Q_conf)
        else:

            self.Style = None
            self.plotThree.update(x, y, Style=self.Style, labels=self.datalabelqt, x_axis_label=x_axis,
                                  y_axis_label=y_axis, legend=self.legend_box, tucl=Tsq_conf, qucl=Q_conf)
        self.unconditional_commit()

    def _update_class_box(self):
        self.plotThree.clear_plot()
        self._setup_plotThree(self.attr_x, self.attr_y)
        self.scoreplot.clear_plot()
        self._setup_score_plot(self.score_x, self.score_y)

    def _update_legend_box(self):
        self.plotThree.clear_plot()
        self._setup_plotThree(self.attr_x, self.attr_y)
        self.scoreplot.clear_plot()
        self._setup_score_plot(self.score_x, self.score_y)
##Loadings plot
    def _update_selection_component_spin(self):

        self._update_plot_spin_selection(self.Principal_Component)

    def plot_spin_selection(self):
        self._remove_spin_selection()
        loadings = self.loadings[self.valid_loadings]
        if self.Principal_Component > loadings.X.shape[0]:
            data = loadings[len(self.valid_loadings) - 1, :]
            self.Error.loading_not_available()
        else:
            data = loadings[self.Principal_Component-1, :]
        self._plot_spin_sel(data, np.where(self.valid_loadings)[0])
        self.loadingsplot.view_box.add_profiles(data.X)

    def _update_plot_spin_selection(self, component):
        self._remove_spin_selection()
        loadings = self.loadings[self.valid_loadings]
        if len(self.valid_loadings) < (component):
            data = loadings[len(self.valid_loadings)-1, :]
            self.Error.loading_not_available()
        else:
            data = loadings[component-1, :]
            self.Error.loading_not_available(shown=False)
        self._plot_spin_sel(data, np.where(self.valid_loadings)[0])
        self.loadingsplot.view_box.add_profiles(data.X)

    def _remove_spin_selection(self):
        for spin_sel in self.__spin_selection:
            spin_sel.remove_items()
        self.loadingsplot.view_box.remove_profiles()
        self.__spin_selection = []

    def _plot_spin_sel(self, data, indices, index=None):
        color = self.__get_spin_sel_color(index)
        spin_sel = Profilespin_sel(data, indices, color, self.loadingsplot)
        kwargs = self.__get_visibility_flags()
        spin_sel.set_visible_profiles(**kwargs)
        self.__spin_selection.append(spin_sel)

    def __get_spin_sel_color(self, index):

        return QColor(LinePlotStyle.DEFAULT_COLOR)

    def __get_visibility_flags(self):
        return {"show_profiles": self.show_profiles,
        }

    def setup_loadingsplot(self):
        if self.loadings is None:
            return

        ticks = [a.name for a in self.graph_variables]
        self.loadingsplot.getAxis("bottom").set_ticks(ticks)
        self.plot_spin_selection()
        self.loadingsplot.view_box.enableAutoRange()
        self.loadingsplot.view_box.updateAutoRange()

    def check_loadings(self):
        def error(err):
            err()
            self.loadings = None

        self.clear_messages()

        if self.loadings is not None:
            self.graph_variables = [var for var in self.loadings.domain.attributes
                                    if var.is_continuous]

            self.valid_loadings = ~countnans(self.loadings.X, axis=1).astype(bool)
            if len(self.graph_variables) < 1:
                error(self.Error.not_enough_attrs)
            elif not np.sum(self.valid_loadings):
                error(self.Error.no_valid_loadings)
            else:
                if not np.all(self.valid_loadings):
                    self.Information.hidden_instances()
                if len(self.graph_variables) > MAX_FEATURES:
                    self.Information.too_many_features()
                    self.graph_variables = self.graph_variables[:MAX_FEATURES]

    def check_display_options(self):
        self.Warning.no_display_option.clear()
        if self.loadings is not None:
            if not (self.show_profiles):
                self.Warning.no_display_option()

    def _set_input_summary(self):
        summary = len(self.loadings) if self.loadings else self.info.NoInput
        details = format_summary_details(self.loadings) if self.loadings else ""
        self.info.set_input_summary(summary, details)

##Score plot
    def set_attr_from_combo(self):
        self.score_changed()
        self.xy_changed_manually.emit(self.score_x, self.score_y)

    def score_changed(self):
        self._setup_score_plot(self.score_x, self.score_y)
        self.commit()

    def _setup_score_plot(self, x_axis, y_axis):

        self.scoreplot.clear_plot()
        if self._pca is None:
            self.scoreplot.clear_plot()
            return
        x=self._Transformed.X[:,self.domainIndexesScore[str(self.score_x)]]
        y=self._Transformed.X[:,self.domainIndexesScore[str(self.score_y)]]
        classes = self.train_classes.copy()
        if self.testdata is not None:
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

            self.scoreplot.update(x,y, Style=self.PlotStyle, labels=self.datalabel, x_axis_label=x_axis, y_axis_label=y_axis, legend=self.legend_box)
        else:

            self.Style = [
                dict(pen=None, symbolBrush=self.SYMBOLBRUSH[0], symbolPen=self.SYMBOLPEN[0], symbol='o', symbolSize=10,
                    name=classes[i]) for i in range(len(classes))]
            self.scoreplot.update(x, y, Style=self.Style, labels=self.datalabel, x_axis_label=x_axis, y_axis_label=y_axis,legend=self.legend_box)

    def init_score_values(self):
        if self.testdata_box:
            if self.testdata_classes_box == True:
                testlabel = np.full(self.testdata.Y.shape, np.max(self.data.Y) + 1, dtype=int)
            else:
                testlabel = self._testdata_transformed.Y + np.max(self.data.Y) + 1
            self.datalabel = np.hstack((self.data.Y, testlabel))
            datatrans = np.vstack((self._transformed.X, self._testdata_transformed.X))
        else:
            self.datalabel = self.data.Y
            datatrans = self._transformed.X
        explVar = np.array(self._variance_ratio, ndmin=2)

        domain = np.array([f'PC {i + 1}  {str(np.round(explVar[:,i]*100,2))}%'
                              for i in range(datatrans.shape[1])],
                            dtype=object)

        for i in range(len(domain)):
            self.domainIndexesScore[domain[i]] = i

        proposed = [a for a in domain]

        dom = Domain(
            [ContinuousVariable(name, compute_value=lambda _: None)
             for name in proposed],
            metas=None)

        self._Transformed = Table(dom, datatrans, metas=None)
        self.xy_modelScore.set_domain(dom)
        self.score_x = self.xy_modelScore[0] if self.xy_modelScore else None
        self.score_y = self.xy_modelScore[1] if len(self.xy_modelScore) >= 2 \
            else self.score_x

##All plots
    def _update_standardize(self):

        if self.standardize:
            self.centering = False

        if self.standardize is False:
            self.centering = True
        self.fit()
        if self.testdata:
            self._testdata_transformed = self.testdata_transform(self.testdata, self._pca)
        if self.data is None:
            self._invalidate_selection()
        else:
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)
            self.init_score_values()
            self._setup_score_plot(self.score_x, self.score_y)

    def _update_centering(self):

        if self.centering:
            self.standardize = False

        if self.centering is False:
            self.standardize = True
        self.fit()
        if self.testdata:
            self._testdata_transformed = self.testdata_transform(self.testdata, self._pca)
        if self.data is None:
            self._invalidate_selection()
        else:
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)
            self.init_score_values()
            self._setup_score_plot(self.score_x, self.score_y)
            self.commit()

    def _update_testdata_box(self):
        if self.testdata is None:
            self.testdata_box = False
            self.testdata_classes_box = False
        else:
            if self.testdata_box == False:
                self.testdata_classes_box = False
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)
            self.init_score_values()
            self._setup_score_plot(self.score_x, self.score_y)
            self.commit()

    def _update_testdata_classes_box(self):
        if self.testdata is None:
            self.testdata_box = False
            self.testdata_classes_box = False
        else:
            if self.testdata_classes_box == True:
                self.testdata_box = True
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)
            self.init_score_values()
            self._setup_score_plot(self.score_x, self.score_y)
            self.commit()


if __name__ == "__main__":
    from sklearn.model_selection import KFold
    data = Table("iris")
    #data = Table("brown-selected")
    KF = KFold(n_splits=2, shuffle=True, random_state=None)
    KF.get_n_splits(data)
    train_index, test_index = KF.split(data)
    X_train, X_test = data[train_index[0]], data[test_index[0]]
    WidgetPreview(OWPCA).run(set_data=data, set_testdata=X_test)
