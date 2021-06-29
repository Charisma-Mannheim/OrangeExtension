import numbers
import numpy
from orangewidget.gui import tabWidget, createTabPage
from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt
from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.data.util import get_unique_names
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.preprocess import preprocess
from Orange.projection import PCA
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.slidergraph import SliderGraph
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output
from orangecontrib.extension.utils.Transform import UnNormalize, UnCenter, Center, Normalize, DoNothing
from orangecontrib.extension.utils.LoggingDummyFile import PrinLog
from orangecontrib.extension.utils import ControlChart
from sklearn.metrics import mean_squared_error
from Orange.widgets.utils.itemmodels import DomainModel
from math import sqrt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import KFold
import time
from joblib import Parallel, delayed, parallel_backend


MAX_COMPONENTS = 100
LINE_NAMES = ["RMSECV by Eigen", "RMSECV row-wise"]
LINE_NAMES_TWO = ["component variance", "cumulative variance"]


class OWPCA(widget.OWWidget):

    name = "PCA"
    description = "Principal component analysis with a diagram of the root mean square error of reconstruction and prediction as well as a scree plot of explained variance."
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
        #components = Output("Loadings", Table)
        #explained_ratio = Output("Explained variance", Table)
        #CumSum = Output("Explained variance cumulative", Table)
        #rmseCV = Output("Reconstruction error of cross validation row-wise", Table)
        #RMSECV = Output("Reconstruction error of cross validation by Eigenvector", Table)
        #outlier = Output("Outlier", Table)
        #inlier = Output("Outlier corrected Dataset", Table)
        #pca = Output("PCA", PCA, dynamic=False)


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

    graph_name = "plot.plotItem"

    class Warning(widget.OWWidget.Warning):
        trivial_components = widget.Msg(
            "All components of the PCA are trivial (explain 0 variance). "
            "Input data is constant (or near constant).")

    class Error(widget.OWWidget.Error):
        no_features = widget.Msg("At least 1 feature is required")
        no_instances = widget.Msg("At least 1 data instance is required")
        no_traindata = widget.Msg("No train data submitted")

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
        self._RMSECV = None
        self._rmseCV = None
        self._statistics = None
        self.train_classes = None
        self.train_datalabel = None
        self.classes = None
        self.outlier_metas = None
            #numpy.empty((1,1))
        self.inlier_data = None
        self.outlier_ids = None

        self.SYMBOLBRUSH = [(0, 204, 204, 180), (51, 255, 51, 180), (255, 51, 51, 180), (0, 128, 0, 180), (19, 234, 201, 180), \
                       (195, 46, 212, 180), (250, 194, 5, 180), (55, 55, 55, 180), (0, 114, 189, 180), (217, 83, 25, 180), (237, 177, 32, 180), \
                       (126, 47, 142, 180), (119, 172, 180)]

        self.SYMBOLPEN = [(0, 204, 204, 255), (51, 255, 51, 255), (255, 51, 51, 255), (0, 128, 0, 255), (19, 234, 201, 255), \
                       (195, 46, 212, 255), (250, 194, 5, 255), (55, 55, 55, 255), (0, 114, 189, 255), (217, 83, 25, 255), (237, 177, 32, 255), \
                       (126, 47, 142, 255), (119, 172, 255)]

        self._init_projector()

        box = gui.vBox(self.controlArea, "Components Selection")
        form = QFormLayout()
        box.layout().addLayout(form)

        self.components_spin = gui.spin(
            box, self, "ncomponents", 1, MAX_COMPONENTS,
            callback=self._update_selection_component_spin,
            keyboardTracking=False
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

        class_box = gui.vBox(self.controlArea, "Control chart options")

        self.classb = gui.checkBox(class_box,
            self, value="class_box", label="Color by class",
            callback=self._update_class_box, tooltip="Datapoints get colored by class, when checked")

        self.legendb = gui.checkBox(class_box,
            self, value="legend_box", label="Show legend",
            callback=self._update_legend_box, tooltip=None)

        gui.comboBox(
            class_box, self, "confidence", label="Shown level of confidence: ",
            items=[str(x) for x in self.Clvl],
            orientation=Qt.Horizontal, callback=self._param_changed)


        self.controlArea.layout().addStretch()
        gui.auto_apply(self.controlArea, self, "auto_commit")


        self.plot = SliderGraph(
            "Principal Components", "RMSECV",
            self._on_cut_changed)
        self.plotTwo = SliderGraph(
            "Principal Components", "Proportion of variance",
            self._on_cut_changed_two)

        self.plotThree = ControlChart.ScatterGraph(callback=None)

        tabs = tabWidget(self.mainArea)

        # graph tab
        tab = createTabPage(tabs, "Error")
        tab.layout().addWidget(self.plot)

        # table tab
        tab = createTabPage(tabs, "Scree")
        tab.layout().addWidget(self.plotTwo)

        tab = createTabPage(tabs, "Q residuals vs. Hotellings T²")
        tab.layout().addWidget(self.plotThree)
        #self._update_standardize()
        self._update_centering()

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
                self.classes = numpy.arange(0, len(self.data.domain.class_var.values))
                self.train_classes = {int(self.classes[i]): self.data.domain.class_var.values[i] for i in
                                      numpy.arange(0, len(self.data.domain.class_var.values))}
            else:
                self.classes = None
            self.train_datalabel = self.data.Y
            self.fit()
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)
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

        self.unconditional_commit()

    def testdata_transform(self, data, projector):

        X = data
        Projector = projector
        transformed = Projector(X)
        return transformed

    def init_attr_values(self):

        X = self.data
        colMeans = numpy.mean(X.X,0)
        colSD = numpy.std(X.X,0)
        n = self.ncomponents

        if self.standardize == True and self.centering == False:
            X_preprocessed = Normalize()(X.X, colMeans, colSD)
            XArray = numpy.copy(X_preprocessed)
        elif self.centering == True and self.standardize == False:
            X_preprocessed = X.X
            XArray = numpy.copy(X_preprocessed)

        else:
            return

        pca = sklearnPCA(n_components=n, random_state=42, svd_solver='full')
        pca.fit(XArray)
        Xtransformed = pca.transform(XArray)
        T = Xtransformed
        P = self._pca.components_[:n, :]
        X_pred = numpy.dot(T, P) + pca.mean_

        Err = XArray - X_pred
        Q = numpy.sum(Err ** 2, axis=1)
        Q = Q.reshape((len(Q),1))

        # Calculate Hotelling's T-squared (note that data are normalised by default)
        Tsq = numpy.sum((T / numpy.std(T, axis=0)) ** 2, axis=1)
        Tsq = Tsq.reshape((len(Tsq),1))

        statistics = numpy.hstack((Tsq, Q))


        domain = numpy.array(['T²', 'Q'],
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

        # set the confidence level
        conf = self.Clvl[self.confidence]/100
        from scipy.stats import f
        # Calculate confidence level for T-squared from the ppf of the F distribution
        Tsq_conf = f.ppf(q=conf, dfn=self.ncomponents,dfd=self.data.X.shape[0]) * self.ncomponents * (self.data.X.shape[0] - 1) / (self.data.X.shape[0] - self.ncomponents)
        # Estimate the confidence level for the Q-residuals
        Qsorted = numpy.sort(y)
        i = len(Qsorted)-1

        while 1 - numpy.sum(y > Qsorted[i]) / numpy.sum(y > 0) > conf:
            i -= 1
            if i == 0:
                break
        Q_conf = Qsorted[i]




        data = self.data[:]
        outlier_index_Q = y > Q_conf
        outlier_index_T = x > Tsq_conf
        outlier_index = outlier_index_Q + outlier_index_T
        inlier_index = ~outlier_index

        if data.metas.shape[1] == 0:
            data.metas = numpy.array(['Sample {}'.format(i) for i in range(data.metas.shape[0])])
            data.metas = data.metas.reshape(len(data.metas),1)
            self.outlier_metas = data.metas[outlier_index]
            self.outlier_ids = numpy.array([i for i in range(data.metas.shape[0])])[outlier_index]
        else:
            self.outlier_ids = numpy.array([i for i in range(data.X.shape[0])])[outlier_index]
            outlier_index = outlier_index.reshape(len(outlier_index),1)
            self.outlier_metas = numpy.array(data.metas[outlier_index])
            self.outlier_metas = self.outlier_metas.reshape((len(self.outlier_metas), 1))
           # self.outlier_ids = numpy.array([i for i in range(len(data.metas))])[outlier_index]


            #ids = numpy.array([i for i in range(data.X.shape[0])])[outlier_index]
            #outlier_index = outlier_index.reshape(len(outlier_index),1)
            #outlier_metas = data.metas[outlier_index]
            #outlier_metas = outlier_metas.reshape(len(outlier_metas),1)


        inlier_data = data
        inlier_data.X = data.X[inlier_index, :]
        inlier_data.W = data.W[inlier_index, :]
        inlier_data.Y = data.Y[inlier_index]
        inlier_data.ids = data.ids[inlier_index]
        inlier_data.metas = data.metas[inlier_index]
        self.inlier_data = inlier_data



        if self.classes is not None:

            if self.class_box:

                self.PlotStyle = [
                    dict(pen=None, symbolBrush=self.SYMBOLBRUSH[i], symbolPen=self.SYMBOLPEN[i], symbol='o', symbolSize=10,
                        name=self.train_classes[i]) for i in range(len(self.train_classes))]

                self.plotThree.update(x,y, Style=self.PlotStyle, labels=self.train_datalabel, x_axis_label=x_axis, y_axis_label=y_axis, legend=self.legend_box, tucl=Tsq_conf, qucl=Q_conf)
            else:

                self.Style = [
                    dict(pen=None, symbolBrush=self.SYMBOLBRUSH[0], symbolPen=self.SYMBOLPEN[0], symbol='o', symbolSize=10,
                        name=self.train_classes[i]) for i in range(len(self.train_classes))]
                self.plotThree.update(x, y, Style=self.Style, labels=self.train_datalabel, x_axis_label=x_axis, y_axis_label=y_axis,legend=self.legend_box, tucl=Tsq_conf, qucl=Q_conf)
        else:

            self.Style = None
            self.plotThree.update(x, y, Style=self.Style, labels=self.train_datalabel, x_axis_label=x_axis,
                                  y_axis_label=y_axis, legend=self.legend_box, tucl=Tsq_conf, qucl=Q_conf)
        self.unconditional_commit()

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
            cumulative = numpy.cumsum(variance_ratio)
            if len(pca.components_) >= 30:
                COMPONENTS = 30
            else:
                COMPONENTS = pca.components_.shape[1]
            pb = gui.ProgressBar(self, 10)
            pbtwo = gui.ProgressBar(self, 10)
            rmseCV, RMSECV = self.rmseCV(Data, COMPONENTS, pb=[pb,pbtwo])
            pb.finish()
            pbtwo.finish()
            if numpy.isfinite(cumulative[-1]):
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

    def clear(self):

        self._pca = None
        self._transformed = None
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
        self.Outputs.explained_ratio.send(None)
        self.Outputs.CumSum.send(None)
        self.Outputs.RMSE.send(None)
        self.Outputs.RMSECV.send(None)
        self.Outputs.rmseCV.send(None)
        self.Outputs.pca.send(self._pca_projector)
        self.Outputs.outlier.send(None)
        self.Outputs.inlier.send(None)

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
            numpy.arange(1, p+1), [RMSECV[:p,0], rmseCV[:p,0]],
            [Qt.blue, Qt.red], cutpoint_x=cutpos, names=LINE_NAMES)
        b = [rmseCV, RMSECV]
        #self.plot.setRange(yRange=(min(RMSECV[:p,0]), max(RMSECV[:p,0])))
        self.plot.setRange(yRange=(min(yi.min() for yi in [rmseCV, RMSECV]), max(yi.max() for yi in [rmseCV, RMSECV])))
        explained_ratio = self._variance_ratio
        explained = self._cumulative
        cutposTwo = self._nselected_components_two()
        pTwo = min(len(self._variance_ratio), self.maxp)

        self.plotTwo.update(
            numpy.arange(1, p+1), [explained_ratio[:pTwo], explained[:pTwo]],
            [Qt.red, Qt.darkYellow], cutpoint_x=cutposTwo, names=LINE_NAMES_TWO)

        self._update_axis()

    def _on_cut_changed(self, components):

        if components == self.ncomponents \
                or self.ncomponents == 0 \
                or self._pca is not None \
                and components == len(self._RMSECV):
            return
        self.ncomponents = components
        if self._pca is not None:
            var = self._RMSECV[components - 1]
            if numpy.isfinite(var):
                self.RMSECV = int(var)
        self._invalidate_selection()

    def _on_cut_changed_two(self, components):

        if components == self.ncomponents \
                or self.ncomponents == 0 \
                or self._pca is not None \
                and components == len(self._variance_ratio):
            return
        self.ncomponents = components
        if self._pca is not None:
            var = self._cumulative[components - 1]
            if numpy.isfinite(var):
                self.variance_covered = int(var * 100)
        self._invalidate_selection()

    def _update_selection_component_spin(self):

        if self._pca is None:
            self._invalidate_selection()
            return

        if self.ncomponents == 0:

            cut = len(self._RMSE)

        else:
            cut = self.ncomponents

        var = self._RMSECV[cut - 1]

        if numpy.isfinite(var):
            self.RMSECV = int(var)
        self.ncomponents = cut
        self.plot.set_cut_point(cut)
        self.plotTwo.set_cut_point(cut)
        self.init_attr_values()
        self._setup_plotThree(self.attr_x, self.attr_y)
        self._invalidate_selection()

    def _update_selection_variance_spin(self):

        if self._pca is None:
            return
        cut = numpy.searchsorted(self._cumulative,
                                 self.variance_covered / 100.0) + 1
        cut = min(cut, len(self._cumulative))
        self.ncomponents = cut
        self.plot.set_cut_point(cut)
        self.plotTwo.set_cut_point(cut)
        self.init_attr_values()
        self._setup_plotThree(self.attr_x, self.attr_y)
        self._invalidate_selection()

    def preprocess(data, preprocessors):

        for pp in preprocessors:
            data = pp(data)
        return data

    def rmseCV(self, data, ncomponents, pb):
        X = data
        colMeans = numpy.mean(X.X,0)
        colSD = numpy.std(X.X,0)
        n = ncomponents

        if self.standardize == True and self.centering == False:
            X_preprocessed = Normalize()(X.X, colMeans, colSD)
            XArray = numpy.copy(X_preprocessed)
        elif self.centering == True and self.standardize == False:
            X_preprocessed = Center()(X.X, colMeans)
            XArray = numpy.copy(X_preprocessed)

        n_splits = 10
        Kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmseCV_list_matrix = []
        for train_index, test_index in Kf.split(XArray):
            X_train, X_test = XArray[train_index], XArray[test_index]
            pca = sklearnPCA(random_state=42, svd_solver='full')
            pca.fit(X_train)
            def innerfunctwo(index):
                X_testtransformed = numpy.dot(X_test, pca.components_.T[:,:index])
                X_test_pred = numpy.dot(X_testtransformed, pca.components_[:index,:])
                #rmseCV = mean_squared_error(X_test, X_test_pred, squared=False)
                rmseCV = numpy.sum((X_test - X_test_pred)**2)
                return rmseCV

            rmseCV_list = [innerfunctwo(index=index) for index in range(1,n+1,1)]
            rmseCVarray = numpy.array(rmseCV_list)
            rmseCV_list_matrix.append(rmseCVarray)
            pb[1].advance()

        #pb.advance()
        rmseCVMatrix = numpy.array(rmseCV_list_matrix)
        rmseCV = numpy.sum(rmseCVMatrix,0)
        rmseCV = numpy.sqrt(rmseCV/(XArray.shape[0]*XArray.shape[1]))
        rmseCV = rmseCV/rmseCV.max()
        rmseCVrowwise = rmseCV.reshape((len(rmseCV), 1))


        KF = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        i = 0
        for train_index, test_index in KF.split(XArray):
            X_train, X_test = XArray[train_index], XArray[test_index]
            pcatwo = sklearnPCA(random_state=42, svd_solver='full')
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
                    t_reduced = numpy.dot(X_test_reduced, P_reduced)
                    #t_reduced = numpy.dot(numpy.dot(X_test_reduced, P_reduced),
                                          #numpy.linalg.inv(numpy.dot(P_reduced.T, P_reduced)))
                    X_test_rest_pred = numpy.dot(t_reduced, P_rest.T)
                    residual_list.append(X_test_rest-X_test_rest_pred)
                c = 0
                for cols in residual_list:
                    if c == 0:
                        residuals = cols
                        c = c + 1
                    else:
                        residuals = numpy.hstack((residuals, cols))
                return residuals
            ResidualsPerComp_list = [innerfunctwo(index=index) for index in range(1,n+1,1)]
            k = 0
            if i == 0:
                RMSECV = numpy.empty((n, n_splits))
                for mat in ResidualsPerComp_list:
                    #RMSECV[k,i] = (numpy.sum(mat ** 2) / (mat.shape[0] * mat.shape[1])) ** 0.5
                    RMSECV[k,i] = numpy.sum(mat ** 2)
                    k = k + 1
                i = i + 1
            else:
               # id = 0
                #for mat, matneu in zip(ResidualsPerComp,ResidualsPerComp_list):
                    #ResidualsPerComp[id] = mat + matneu
                    #ResidualsPerComp[id] = numpy.vstack((mat, matneu))
                    #id = id +1


                for mat in ResidualsPerComp_list:
                    #RMSECV[k,i] = (numpy.sum(mat ** 2) / (mat.shape[0] * mat.shape[1])) ** 0.5
                    RMSECV[k, i] = numpy.sum(mat ** 2)
                    k = k + 1
                i = i + 1
            pb[0].advance()
        RMSECV = numpy.sum(RMSECV, axis=1)
        RMSECV = numpy.sqrt(RMSECV/(XArray.shape[0]*XArray.shape[1]))
        RMSECV = RMSECV/RMSECV.max()
        rmseCVbyEigen = RMSECV.reshape((len(RMSECV),1))
        return rmseCVrowwise, rmseCVbyEigen

    def _update_standardize(self):

        if self.standardize:
            self.centering = False

        if self.standardize is False:
            self.centering = True
        self.fit()
        if self.data is None:
            self._invalidate_selection()
        else:
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)

    def setProgressValue(self, value):

        self.progressBarSet(value)

    def _update_centering(self):

        if self.centering:
            self.standardize = False

        if self.centering is False:
            self.standardize = True
        self.fit()
        if self.data is None:
            self._invalidate_selection()
        else:
            self.init_attr_values()
            self._setup_plotThree(self.attr_x, self.attr_y)

    def _update_class_box(self):
        self.plotThree.clear_plot()
        self._setup_plotThree(self.attr_x, self.attr_y)

    def _update_legend_box(self):
        self.plotThree.clear_plot()
        self._setup_plotThree(self.attr_x, self.attr_y)

    def _init_projector(self):

        self._pca_projector = PCA(n_components=MAX_COMPONENTS, random_state=0)
        self._pca_projector.component = self.ncomponents
        self._pca_preprocessors = PCA.preprocessors

    def _nselected_components(self):

        """Return the number of selected components."""
        if self._pca is None:
            return 0
        if self.ncomponents == 0:
            max_comp = len(self._RMSECV)
        else:
            max_comp = self.ncomponents
        RMSE_max = self._RMSECV[max_comp - 1]
        #if RMSE_max != numpy.floor(self.RMSECV):
        cut = max_comp
        assert numpy.isfinite(RMSE_max)
        self.RMSECV = int(RMSE_max)
        #else:
            #self.ncomponents = cut = numpy.searchsorted(
                #self._RMSECV, self.RMSECV) + 1


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
        if var_max != numpy.floor(self.variance_covered / 100.0):
            cut = max_comp
            assert numpy.isfinite(var_max)
            self.variance_covered = int(var_max * 100)
        else:
            self.ncomponents = cut = numpy.searchsorted(
                self._cumulative, self.variance_covered / 100.0) + 1
        return cut

    def _invalidate_selection(self):
        self.commit()

    def _update_axis(self):
        p = min(len(self._RMSECV), self.maxp)
        axis = self.plot.getAxis("bottom")
        d = max((p-1)//(self.axis_labels-1), 1)
        axis.setTicks([[(i, str(i)) for i in range(1, p + 1, d)]])

    def commit(self):
        inlier = outlier = RMSECV = rmseCV = transformed = transformed_testdata = data = components = CumSum = explVar = None
        if self._pca is not None:
            if self._transformed is None:
                self._transformed = self._pca(self.data)

            transformed = self._transformed
            explVar = numpy.array(self._variance_ratio, ndmin=2)
            explVarFlo = [k for k in explVar]
            a = [str(i)  for i in transformed.domain.attributes[:self.ncomponents]]
            b = [" (" + str(numpy.around(i*100, decimals=2)) + "%) " for i in explVarFlo[0][:self.ncomponents]]
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
            proposed2 = [b.name for b in self._pca.domain.attributes[:self.ncomponents]]
            meta_name2 = get_unique_names(proposed2, 'variance')
            proposed3 = [c.name for c in self._pca.domain.attributes[:self._COMPONENTS]]
            meta_name3 = get_unique_names(proposed3, 'components')

            dom = Domain(
                [ContinuousVariable(name, compute_value=lambda _: None)
                 for name in proposed],
                metas=[StringVariable(name=meta_name)])
            metas = numpy.array([['PC{}'.format(i + 1)
                                  for i in range(self.ncomponents)]],
                                dtype=object).T

            dom2 = Domain(
                [ContinuousVariable(name, compute_value=lambda _: None)
                 for name in proposed2],
                metas=[StringVariable(name=meta_name2)])

            dom3 = Domain(
                [ContinuousVariable(name, compute_value=lambda _: None)
                 for name in proposed3],
                metas=[StringVariable(name=meta_name3)])

            metas2 = numpy.array([['Cumulative Variance'
                                  ]],
                                dtype=object)
            metas3 = numpy.array([['explained Variance'
                                   ]],
                                 dtype=object)

            metas4 = numpy.array([['id'
                                   ]],
                                 dtype=object)

            metas5 = numpy.array([['RMSECV'
                                   ]],
                                 dtype=object)


            if self.outlier_metas is not None:
                proposed4 = [name for name in list(self.outlier_metas)]
                meta_name4 = 'Samples'
                dom8 = Domain(
                    [ContinuousVariable(str(name), compute_value=lambda _: None)
                    for name in proposed4],
                    metas=[StringVariable(name=meta_name4)])
                outlier = numpy.array(self.outlier_ids, ndmin = 2)
                outlier = Table(dom8, outlier, metas=metas4)
                outlier.name = 'Sample info'

            components = Table(dom, self._pca.components_[:self.ncomponents,:],
                               metas=metas)
            components.name = 'components'

            CumSumVar = numpy.array(self._cumulative, ndmin = 2)
            CumSum = Table(dom2, CumSumVar[:,:self.ncomponents], metas=metas2)
            CumSum.name = 'CumSum'

            explVar = numpy.array(self._variance_ratio, ndmin = 2)
            explVar = Table(dom2, explVar[:,:self.ncomponents], metas=metas3)
            explVar.name = 'explVar'
            data_dom = Domain(
                self.data.domain.attributes,
                self.data.domain.class_vars,
                self.data.domain.metas + domain.attributes)
            data = Table.from_numpy(
                data_dom, self.data.X, self.data.Y,
                numpy.hstack((self.data.metas, transformed.X)),
                ids=self.data.ids)
            if self._RMSECV is not None:
                RMSECV = self._RMSECV
                self._RMSECV = RMSECV
                RMSECV = numpy.transpose(RMSECV)
                RMSECV = Table(dom3, RMSECV[:,:self._COMPONENTS], metas=metas5)
                RMSECV.name = 'RMSECV by Eigenvector'
            if self._rmseCV is not None:
                rmseCV = self._rmseCV
                self._rmseCV = rmseCV
                rmseCV = numpy.transpose(rmseCV)
                rmseCV = Table(dom3, rmseCV[:,:self._COMPONENTS], metas=metas5)
                rmseCV.name = 'RMSECV row wise'

            inlier = self.inlier_data
        self._pca_projector.component = self.ncomponents
        self.Outputs.data.send(data)
        self.Outputs.transformed_data.send(transformed)
        self.Outputs.transformed_testdata.send(transformed_testdata)
        #self.Outputs.components.send(components)
        #self.Outputs.explained_ratio.send(explVar)
        #self.Outputs.CumSum.send(CumSum)
        #self.Outputs.RMSECV.send(RMSECV)
        #self.Outputs.rmseCV.send(rmseCV)
        #self.Outputs.pca.send(self._pca_projector)
        #self.Outputs.outlier.send(outlier)
        #self.Outputs.inlier.send(inlier)

    def send_report(self):

        if self.data is None:
            return
        self.report_plot("Reconstruction Error", self.plot)
        self.report_plot("Explained Variance", self.plotTwo)

    def setup_plot(self):
        super().setup_plot()

    @classmethod
    def migrate_settings(cls, settings, version):
        if "variance_covered" in settings:
            # Due to the error in gh-1896 the variance_covered was persisted
            # as a NaN value, causing a TypeError in the widgets `__init__`.
            vc = settings["variance_covered"]
            if isinstance(vc, numbers.Real):
                if numpy.isfinite(vc):
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

if __name__ == "__main__":
    import numpy
    from sklearn.model_selection import KFold
    data = Table("iris")
    #data = Table("brown-selected")
    KF = KFold(n_splits=2, shuffle=False, random_state=None)
    KF.get_n_splits(data)
    train_index, test_index = KF.split(data)
    X_train, X_test = data[train_index[0]], data[test_index[0]]
    WidgetPreview(OWPCA).run(set_data=data, set_testdata=X_test)
