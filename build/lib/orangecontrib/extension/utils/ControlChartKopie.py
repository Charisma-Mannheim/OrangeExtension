from AnyQt.QtGui import QColor
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph import functions as fn
from pyqtgraph import getConfigOption, LegendItem
import numpy as np
from pyqtgraph import PlotWidget, mkPen, InfiniteLine, PlotCurveItem, \
    TextItem, Point
from AnyQt.QtCore import Qt


from xml.sax.saxutils import escape

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QSize, QLineF, pyqtSignal as Signal
from AnyQt.QtGui import QPainter, QPen, QColor
from AnyQt.QtWidgets import QApplication, QGraphicsLineItem

import pyqtgraph as pg
from pyqtgraph.functions import mkPen
from pyqtgraph.graphicsItems.ViewBox import ViewBox

from Orange.data import Table, DiscreteVariable
from Orange.data.sql.table import SqlTable
from Orange.statistics.util import countnans, nanmean, nanmin, nanmax, nanstd
from Orange.widgets import gui, report
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

from pyqtgraph.widgets.ColorMapWidget import ColorMapParameter
import pyqtgraph as pg



__all__ = ['ScatterGraphWidget']


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

class LinePlotViewBox(ViewBox):
    selection_changed = Signal(np.ndarray)

    def __init__(self):
        super().__init__(enableMenu=False)
        self._profile_items = None
        self._can_select = True
        self._graph_state = SELECT

        self.setMouseMode(self.PanMode)

        pen = mkPen(LinePlotStyle.SELECTION_LINE_COLOR,
                    width=LinePlotStyle.SELECTION_LINE_WIDTH)
        self.selection_line = QGraphicsLineItem()
        self.selection_line.setPen(pen)
        self.selection_line.setZValue(1e9)
        self.addItem(self.selection_line, ignoreBounds=True)

    def update_selection_line(self, button_down_pos, current_pos):
        p1 = self.childGroup.mapFromParent(button_down_pos)
        p2 = self.childGroup.mapFromParent(current_pos)
        self.selection_line.setLine(QLineF(p1, p2))
        self.selection_line.resetTransform()
        self.selection_line.show()

    def set_graph_state(self, state):
        self._graph_state = state

    def enable_selection(self, enable):
        self._can_select = enable

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

    def mouseDragEvent(self, event, axis=None):
        if self._graph_state == SELECT and axis is None and self._can_select:
            event.accept()
            if event.button() == Qt.LeftButton:
                self.update_selection_line(event.buttonDownPos(), event.pos())
                if event.isFinish():
                    self.selection_line.hide()
                    p1 = self.childGroup.mapFromParent(
                        event.buttonDownPos(event.button()))
                    p2 = self.childGroup.mapFromParent(event.pos())
                    self.selection_changed.emit(self.get_selected(p1, p2))
        elif self._graph_state == ZOOMING or self._graph_state == PANNING:
            event.ignore()
            super().mouseDragEvent(event, axis=axis)
        else:
            event.ignore()

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

class LinePlotStyle:
    DEFAULT_COLOR = QColor(Qt.blue)
    SELECTION_LINE_COLOR = QColor(Qt.black)
    SELECTION_LINE_WIDTH = 2

    UNSELECTED_LINE_ALPHA = 170
    UNSELECTED_LINE_ALPHA_SEL = 170

    SELECTED_LINE_WIDTH = 2
    SELECTED_LINE_ALPHA = 170

    RANGE_ALPHA = 25
    SELECTED_RANGE_ALPHA = 50

    MEAN_WIDTH = 3
    MEAN_DARK_FACTOR = 110

class ScatterGraphWidget(QtGui.QSplitter):

    def __init__(self, parent=None):
        QtGui.QSplitter.__init__(self, QtCore.Qt.Horizontal)
        self.view_box = LinePlotViewBox()
        self.plot = PlotWidget(background="w", viewBox=self.view_box)
        self.plot.view_box = self.view_box
        self.addWidget(self.plot)
        bg = fn.mkColor(getConfigOption('background'))
        bg.setAlpha(150)
        self.filterText = TextItem(border=getConfigOption('foreground'), color=bg)
        self.filterText.setPos(60, 20)
        self.filterText.setParentItem(self.plot.plotItem)

        self.data = None
        self.valid_data = None
        self.mouseOverField = None
        self.scatterPlot = None
        self.style = dict(pen=None, symbol='o', symbolPen=(0,0,200), symbolSize=10,symbolBrush=(0,65,200), name="data")
        self.selection = set()
        self.view_box.set_graph_state(SELECT)
        self.view_box.setMouseMode(self.view_box.RectMode)

    def updatePlot(self):
        self.plot.clear()
        if self.data is None:
            return

        if self.filtered is None:
            self.filtered = self.filter.filterData(self.data)
        data = self.filtered
        if len(data) == 0:
            return

        colors = np.array([fn.mkBrush(*x) for x in self.colorMap.map(data)])

        style = self.style.copy()

        ## Look up selected columns and units
        sel = list([str(item.text()) for item in self.fieldList.selectedItems()])
        units = list([item.opts.get('units', '') for item in self.fieldList.selectedItems()])
        if len(sel) == 0:
            self.plot.setTitle('')
            return

        if len(sel) == 1:
            self.plot.setLabels(left=('N', ''), bottom=(sel[0], units[0]), title='')
            if len(data) == 0:
                return
            # x = data[sel[0]]
            # y = None
            xy = [data[sel[0]], None]
        elif len(sel) == 2:
            self.plot.setLabels(left=(sel[1], units[1]), bottom=(sel[0], units[0]))
            if len(data) == 0:
                return

            xy = [data[sel[0]], data[sel[1]]]

        enum = [False, False]
        for i in [0, 1]:
            axis = self.plot.getAxis(['bottom', 'left'][i])
            if xy[i] is not None and (
                    self.fields[sel[i]].get('mode', None) == 'enum' or xy[i].dtype.kind in ('S', 'O')):
                vals = self.fields[sel[i]].get('values', list(set(xy[i])))
                xy[i] = np.array([vals.index(x) if x in vals else len(vals) for x in xy[i]], dtype=float)
                axis.setTicks([list(enumerate(vals))])
                enum[i] = True
            else:
                axis.setTicks(None)  # reset to automatic ticking

        ## mask out any nan values
        mask = np.ones(len(xy[0]), dtype=bool)
        if xy[0].dtype.kind == 'f':
            mask &= ~np.isnan(xy[0])
        if xy[1] is not None and xy[1].dtype.kind == 'f':
            mask &= ~np.isnan(xy[1])

        xy[0] = xy[0][mask]
        style['symbolBrush'] = colors[mask]

        ## Scatter y-values for a histogram-like appearance
        if xy[1] is None:
            ## column scatter plot
            xy[1] = fn.pseudoScatter(xy[0])
        else:
            ## beeswarm plots
            xy[1] = xy[1][mask]
            for ax in [0, 1]:
                if not enum[ax]:
                    continue
                imax = int(xy[ax].max()) if len(xy[ax]) > 0 else 0
                for i in range(imax + 1):
                    keymask = xy[ax] == i
                    scatter = fn.pseudoScatter(xy[1 - ax][keymask], bidir=True)
                    if len(scatter) == 0:
                        continue
                    smax = np.abs(scatter).max()
                    if smax != 0:
                        scatter *= 0.2 / smax
                    xy[ax][keymask] += scatter

        if self.scatterPlot is not None:
            try:
                self.scatterPlot.sigPointsClicked.disconnect(self.plotClicked)
            except:
                pass
        self.scatterPlot = self.plot.plot(xy[0], xy[1], data=data[mask], **style)
        self.scatterPlot.sigPointsClicked.connect(self.plotClicked)

    def plotClicked(self, plot, points):
        pass

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
        self.legend.hide()



class ScatterGraph(ScatterGraphWidget):
    enable_selection = Signal(bool)
    selection = Setting(None, schema_only=True)
    show_profiles = Setting(True)
    def __init__(self, callback):
        super().__init__()
        self.__groups = []
        self.group_var = None
        self.__pending_selection = self.selection
        self.plot.plotItem.getViewBox().setMenuEnabled(False)
        self.plot.plotItem.getViewBox().setMouseEnabled(False, False)
        self.plot.plotItem.showGrid(True, True, alpha=0.5)
        self.plot.plotItem.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0))

        self.callback = callback
        self.sequences = None
        self.selection = set()
        self._linet = None
        self._lineq = None
        self.x = None
        self.plot.view_box.selection_changed.connect(self.selection_changed)
        self.enable_selection.connect(self.plot.view_box.enable_selection)
        self.selected = None

    def update(self, x, y, Style = None, labels=None, x_axis_label=None, y_axis_label=None, legend=None, tucl = None, qucl = None):
        data = np.vstack((x,y))
        self.data = data.T
        self.valid_data = ~countnans(self.data, axis=1).astype(bool)
        axis = self.plot.plotItem.getAxis("bottom")
        axis.setLabel(x_axis_label)
        axis.setPen(color="k", width=1)
        axis = self.plot.plotItem.getAxis("left")
        axis.setLabel(y_axis_label)
        axis.setPen(color="k", width=1)
        Legend = self.plot.addLegend(offset=(0, 360))
        if Style == None:
            style = self.style
            self.sequences = y
            self.x = x
            s = y

            if legend:
                self.plot.clear(LegendItem)
                LegendItem.setParentItem(self.plot)
                self.plot.plot(x, s ,**style, antialias=True )
                self.plot.plotItem.setRange(xRange=(x.min(), x.max()),
                        yRange=(y.min(), y.max()))

                self.plot.clear(LegendItem)
                self.plot.plot(x, s, **style, antialias=True)
                self.plot.plotItem.setRange(xRange=(x.min(), x.max()),
                                            yRange=(y.min(), y.max()))
        elif isinstance(Style, list):
            self.sequences = y
            self.x = x
            for i in range(len(Style)):
                style = Style[i]
                s = y[labels == i]
                v = x[labels == i]
                if legend:
                    self.plot.plot(v, s, **style, antialias=True)

                else:
                    self.plot.plot(v, s, **style, antialias=True)
                    Legend.clear()
                    self._plot_cutpointT(tucl)
                    self._plot_cutpointQ(qucl)
                    self.plot.plotItem.setRange(xRange=(x.min(), x.max()),
                                                yRange=(y.min(), y.max()))

                self.plot.enableAutoRange()
        else:
            style = Style
            self.sequences = y
            self.x = x
            s = y
            if legend:
                self.plot.plot(x, s ,**style, antialias=True )
                self.plot.enableAutoRange()
            else:
                self.plot.plot(x, s ,**style, antialias=True )
                Legend.clear()
                self.plot.enableAutoRange()
        self.apply_selection()
        self._plot_cutpointT(tucl)
        self._plot_cutpointQ(qucl)

    def apply_selection(self):
        if self.data is not None and self.__pending_selection is not None:
            sel = [i for i in self.__pending_selection if i < len(self.data)]
            mask = np.zeros(len(self.data), dtype=bool)
            mask[sel] = True
            mask = mask[self.valid_data]
            self.selection_changed(mask)
            self.__pending_selection = None

    def _set_anchor(self):
        #, label, cutidx
        #label.anchor = Point(0, 1) if cutidx < len(self.x) / 2 \
        #else
        Point(1, 0)

    def selection_changed(self, mask):
        if self.data is None:
            return
        # need indices for self.data: mask refers to self.data[self.valid_data]
        indices = np.arange(len(self.data))[self.valid_data][mask]
        self.select(indices)
        old = self.selection
        self._plot_group(data=self.data, indices=indices, index=None)
        if not old and self.selection or old and not self.selection:
            self._update_profiles_color()
        self._update_sel_profiles_and_range()
        self._update_sel_profiles_color()
    def __get_group_color(self, index):
        if self.group_var is not None:
            return QColor(*self.group_var.colors[index])
        return QColor(LinePlotStyle.DEFAULT_COLOR)
    def _plot_group(self, data, indices, index=None):
        color = self.__get_group_color(index)
        group = ProfileGroup(data, indices, color, self.plot)
        #kwargs = self.__get_visibility_flags()
        #group.set_visible_profiles(**kwargs)
        self.__groups.append(group)

    def __get_visibility_flags(self):
        return {"show_profiles": self.show_profiles}


    def _update_sel_profiles_and_range(self):
        # mark selected instances and selected range
        for group in self.__groups:
            inds = [i for i in group.indices if self.__in(i, self.selection)]
            table = self.data[inds, self.graph_variables].X if inds else None
            #group.update_sel_profiles(table)
    def update_sel_profiles(self, y_data):
        x, y, connect = self.__get_disconnected_curve_data(y_data) \
            if y_data is not None else (None, None, None)
        self.sel_profiles.setData(x=x, y=y, connect=connect)

    def _update_sel_profiles_color(self):
        # color depends on subset; when subset is present,
        # selected profiles are black
        if not self.selection or not self.show_profiles:
            return
        for group in self.__groups:
            group.update_sel_profiles_color(bool(self.subset_indices))

    def clear_plot(self):

        self.plot.clear()
        self.sequences = None

    def _plot_cutpointT(self, x):
        """
        Function plots the cutpoint.

        Parameters
        ----------
        x : int
            Cutpoint location.
        """
        self._linet = None
        if x is None:
            self._linet = None
            return
        if self._linet is None:
            # plot interactive vertical line
            self._linet = InfiniteLine(
                angle=90, pos=x, movable=False,
                bounds=(self.x.min(), self.x.max())
            )
            self._linet.setPen(mkPen(QColor(Qt.blue), style=Qt.DashLine, width=1))
            self.plot.addItem(self._linet)
        else:
            self._linet is None
            self._linet.setValue(x)
            self.plot.addItem(self._linet)

        #self._update_horizontal_lines()

    def _plot_cutpointQ(self, x):
        """
        Function plots the cutpoint.

        Parameters
        ----------
        x : int
            Cutpoint location.
        """
        self._lineq = None
        if x is None:
            self._lineq = None
            return
        if self._lineq is None:
            # plot interactive vertical line
            self._lineq = InfiniteLine(
                angle=0, pos=x, movable=False,
                bounds=(self.sequences.min(), self.sequences.max())
            )
            self._lineq.setPen(mkPen(QColor(Qt.blue), style=Qt.DashLine, width=1))
            self.plot.addItem(self._lineq)
        else:
            self._lineq.setValue(x)
            self.plot.addItem(self._lineq)

        #self._update_horizontal_lines()

    def set_cut_pointT(self, x):
        """
        This function sets the cutpoint (selection line) at the specific
        location.

        Parameters
        ----------
        x : int
            Cutpoint location at the x axis.
        """
        self._plot_cutpointT(x)

    def set_cut_pointQ(self, x):
        """
        This function sets the cutpoint (selection line) at the specific
        location.

        Parameters
        ----------
        x : int
            Cutpoint location at the x axis.
        """
        self._plot_cutpointQ(x)

class ProfileGroup:
    def __init__(self, data, indices, color, graph):
        self.x_data = np.arange(1, data.shape[1] + 1)
        self.y_data = data
        self.indices = indices
        self.ids = None
        self.color = color
        self.graph = graph

        self.profiles_added = False
        self.sub_profiles_added = False
        self.range_added = False
        self.mean_added = False
        self.error_bar_added = False

        self.graph_items = []
        self.__mean = nanmean(self.y_data, axis=0)



