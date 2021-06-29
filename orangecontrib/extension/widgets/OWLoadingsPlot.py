from xml.sax.saxutils import escape
import sys
import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QSize, QLineF, pyqtSignal as Signal
from AnyQt.QtGui import QPainter, QPen, QColor
from AnyQt.QtWidgets import QApplication, QGraphicsLineItem
from AnyQt.QtWidgets import QFormLayout

import pyqtgraph as pg
from pyqtgraph.functions import mkPen
from pyqtgraph.graphicsItems.ViewBox import ViewBox

from Orange.data import Table, DiscreteVariable
from Orange.data.sql.table import SqlTable
from Orange.statistics.util import countnans, nanmean, nanmin, nanmax, nanstd
from Orange.widgets import gui
from Orange.widgets.settings import (
    Setting, ContextSetting, DomainContextHandler
)
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.plot import OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.visualize.owdistributions import LegendItem
from Orange.widgets.widget import OWWidget, Input, Output, Msg


def ccw(a, b, c):
    """
    Checks whether three points are listed in a counterclockwise order.
    """
    ax, ay = (a[:, 0], a[:, 1]) if a.ndim == 2 else (a[0], a[1])
    bx, by = (b[:, 0], b[:, 1]) if b.ndim == 2 else (b[0], b[1])
    cx, cy = (c[:, 0], c[:, 1]) if c.ndim == 2 else (c[0], c[1])
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

#Tests
def intersects(a, b, c, d):
    """
    Checks whether line segment a (given points a and b) intersects with line
    segment b (given points c and d).
    """
    return np.logical_and(ccw(a, c, d) != ccw(b, c, d),
                          ccw(a, b, c) != ccw(a, b, d))

#Tests
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

#Cosmetics
class LinePlotStyle:
    DEFAULT_COLOR = QColor(Qt.blue)

    UNSELECTED_LINE_ALPHA = 170

#x- and y-axis
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
    def make_pen(color, width=3):
        pen = QPen(color, width)
        pen.setCosmetic(True)
        return pen


MAX_FEATURES = 10000


class LoadingsPlot(OWWidget):
    name = "Loadings Plot"
    description = "Visualization of contribution from features to Principal Components"
    icon = "icons/LoadingsPlot.svg"
    priority = 2

    class Inputs:
        data = Input("Data", Table, default=True)

    settingsHandler = DomainContextHandler()
    show_profiles = Setting(True)
    auto_commit = Setting(True)
    Principal_Component = Setting(1)
    graph_name = "graph.plotItem"


    class Error(OWWidget.Error):
        not_enough_attrs = Msg("Need at least one continuous feature.")
        no_valid_data = Msg("No plot due to no valid data.")

    class Warning(OWWidget.Warning):
        no_display_option = Msg("No display option is selected.")

    class Information(OWWidget.Information):
        hidden_instances = Msg("Instances with unknown values are not shown.")
        too_many_features = Msg("Data has too many features. Only first {}"
                                " are shown.".format(MAX_FEATURES))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__spin_selection = []
        self.data = None
        self.valid_data = None
        self.subset_data = None
        self.subset_indices = None
        self.graph_variables = []
        self.setup_gui()

        #self.graph.view_box.selection_changed.connect(self.selection_changed)

    def setup_gui(self):
        self._add_graph()
        self._add_controls()

    def _add_graph(self):
        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = LinePlotGraph(self, y_axis_label="Loadings")

        box.layout().addWidget(self.graph)

    def _add_controls(self):

        box = gui.vBox(self.controlArea, "Display Loadings for")
        form = QFormLayout()
        box.layout().addLayout(form)


        self.components_spin = gui.spin(
            box, self, "Principal_Component", 1, MAX_FEATURES,
            callback=self._update_selection_component_spin,
            keyboardTracking=False
        )
        form.addRow("Component:", self.components_spin)

    def _update_selection_component_spin(self):

        self._update_plot_spin_selection(self.Principal_Component)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.closeContext()
        self.data = data
        self._set_input_summary()
        self.clear()
        self.check_data()
        self.check_display_options()
        self.openContext(data)
        self.setup_plot()

    def check_data(self):
        def error(err):
            err()
            self.data = None

        self.clear_messages()
        if self.data is not None:
            self.graph_variables = [var for var in self.data.domain.attributes
                                    if var.is_continuous]

            self.valid_data = ~countnans(self.data.X, axis=1).astype(bool)
            if len(self.graph_variables) < 1:
                error(self.Error.not_enough_attrs)
            elif not np.sum(self.valid_data):
                error(self.Error.no_valid_data)
            else:
                if not np.all(self.valid_data):
                    self.Information.hidden_instances()
                if len(self.graph_variables) > MAX_FEATURES:
                    self.Information.too_many_features()
                    self.graph_variables = self.graph_variables[:MAX_FEATURES]

    def check_display_options(self):
        self.Warning.no_display_option.clear()
        if self.data is not None:
            if not (self.show_profiles):
                self.Warning.no_display_option()

    def _set_input_summary(self):
        summary = len(self.data) if self.data else self.info.NoInput
        details = format_summary_details(self.data) if self.data else ""
        self.info.set_input_summary(summary, details)

    def setup_plot(self):
        if self.data is None:
            return

        ticks = [a.name for a in self.graph_variables]
        #data = self.data[self.valid_data, self.graph_variables]
        #datalist = list(data.X[self.Principal_Component-1, :])
        #ticks2 = [str(a) for a in datalist]
        self.graph.getAxis("bottom").set_ticks(ticks)
        #self.graph.getAxis("left")

        #self.graph.getAxis("left")
        self.plot_spin_selection()
        self.graph.view_box.enableAutoRange()
        self.graph.view_box.updateAutoRange()

    def plot_spin_selection(self):
        self._remove_spin_selection()
        data = self.data[self.valid_data, self.graph_variables]
        data = data[self.Principal_Component-1, :]
        self._plot_spin_sel(data, np.where(self.valid_data)[0])
        self.graph.view_box.add_profiles(data.X)

    def _update_plot_spin_selection(self, component):
        self._remove_spin_selection()
        data = self.data[self.valid_data, self.graph_variables]
        data = data[component-1, :]
        self._plot_spin_sel(data, np.where(self.valid_data)[0])
        self.graph.view_box.add_profiles(data.X)

    def _remove_spin_selection(self):
        for spin_sel in self.__spin_selection:
            spin_sel.remove_items()
        self.graph.view_box.remove_profiles()
        self.__spin_selection = []

    def _plot_spin_sel(self, data, indices, index=None):
        color = self.__get_spin_sel_color(index)
        spin_sel = Profilespin_sel(data, indices, color, self.graph)
        kwargs = self.__get_visibility_flags()
        spin_sel.set_visible_profiles(**kwargs)
        self.__spin_selection.append(spin_sel)

    def __get_spin_sel_color(self, index):

        return QColor(LinePlotStyle.DEFAULT_COLOR)

    def __get_visibility_flags(self):
        return {"show_profiles": self.show_profiles,
        }

    def sizeHint(self):
        return QSize(1132, 708)

    def clear(self):
        self.valid_data = None
        self.selection = None
        self.__spin_selection = []
        self.graph_variables = []
        self.graph.reset()

    @staticmethod
    def __in(obj, collection):
        return collection is not None and obj in collection

if __name__ == "__main__":
    data = Table("housing")
    WidgetPreview(LoadingsPlot).run(set_data=data)
