from AnyQt.QtGui import QColor
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph import functions as fn
from pyqtgraph import getConfigOption, LegendItem
import numpy as np
from pyqtgraph import PlotWidget, mkPen, InfiniteLine, PlotCurveItem, \
    TextItem, Point, GraphicsWidget
from AnyQt.QtCore import Qt

from pyqtgraph.widgets.ColorMapWidget import ColorMapParameter
import pyqtgraph as pg



__all__ = ['ScatterGraphWidget']


class ScatterGraphWidget(QtGui.QSplitter):

    def __init__(self, parent=None):
        QtGui.QSplitter.__init__(self, QtCore.Qt.Horizontal)
        self.plot = PlotWidget(background="w")
        self.addWidget(self.plot)
        bg = fn.mkColor(getConfigOption('background'))
        bg.setAlpha(150)
        self.filterText = TextItem(border=getConfigOption('foreground'), color=bg)
        self.filterText.setPos(60, 20)
        self.filterText.setParentItem(self.plot.plotItem)

        self.data = None
        self.mouseOverField = None
        self.scatterPlot = None
        self.style = dict(pen=None, symbol='o', symbolPen=(0,0,200), symbolSize=10,symbolBrush=(0,65,200), name="data")

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

class ScatterGraph(ScatterGraphWidget, GraphicsWidget):
    def __init__(self, callback):
        super().__init__()
        self.plot.plotItem.getViewBox().setMenuEnabled(False)
        self.plot.plotItem.getViewBox().setMouseEnabled(False, False)
        self.plot.plotItem.showGrid(True, True, alpha=0.5)
        self.plot.plotItem.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0))

        self.callback = callback
        self.sequences = None
        self.x = None
    def update(self, x, y, Style = None, labels=None, x_axis_label=None, y_axis_label=None, legend=None):

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
            else:
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
                    self.plot.enableAutoScale()

                else:
                    self.plot.plot(v, s, **style, antialias=True)
                    Legend.clear()
                    self.plot.plotItem.setRange(xRange=(x.min(), x.max()),
                                                yRange=(y.min(), y.max()))
                    self.plot.enableAutoScale()
        else:
            style = Style
            self.sequences = y
            self.x = x
            s = y
            if legend:
                self.plot.plot(x, s ,**style, antialias=True )
                self.plot.enableAutoScale()
            else:
                self.plot.plot(x, s ,**style, antialias=True )
                Legend.clear()
                self.plot.enableAutoScale()

    def _set_anchor(self):
        #, label, cutidx
        #label.anchor = Point(0, 1) if cutidx < len(self.x) / 2 \
        #else
        Point(1, 0)
    def clear_plot(self):

        self.plot.clear()
        self.sequences = None










