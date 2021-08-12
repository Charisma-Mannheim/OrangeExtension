
from orangecontrib.extension.utils.Regression.linearExtension import PLSRLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Output, Input
from Orange.widgets import settings, gui
from orangecontrib.extension.utils.LoggingDummy import PrinLog
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
import numpy as np
from Orange.data.util import get_unique_names



class OWPLSRegression(OWBaseLearner):
    name = "PLS Regression"
    description = "A partial least square regression algorithm "
    icon = "icons/PLSRegression.svg"
    replaces = [
        "Orange.widgets.regression.owlinearregression.OWLinearRegression",
    ]
    priority = 1
    keywords = ["PLS-R"]

    LEARNER = PLSRLearner

    class Outputs(OWBaseLearner.Outputs):
        coefficients = Output("Coefficients", Table, explicit=True)
        scores = Output("Scores", Table, explicit=True)
        loadings = Output("Loadings", Table, explicit=True)

    #: Types
    ncomponents = settings.Setting(3)
    max_iter = settings.Setting(500)
    autosend = settings.Setting(True)
    MAX_COMPONENTS = 100

    def add_main_layout(self):
        self.data = None

        self.box = gui.vBox(
            self.controlArea, "PLS Parameters")
        self.ncomps_spin = gui.spin(
            self.box, self, "ncomponents", 1, 50, 1,
            label="Components: ", controlWidth=100,
            callback=self.settings_changed)
        self.n_iters = gui.spin(
            self.box, self, "max_iter", 5, 100000, 50,
            label="Iteration limit: ", controlWidth=100,
            callback=self.settings_changed,
            checkCallback=self.settings_changed)

    def handleNewSignals(self):
        self.apply()

    def create_learner(self):

        preprocessors = self.preprocessors
        ncomponents = self.ncomponents
        args = {"preprocessors": preprocessors, "n_components" : ncomponents}
        learner = PLSRLearner(**args)
        return learner

    def update_model(self):
        super().update_model()
        coef_table = None
        loadings_table = None
        scores_table = None
        if self.model is not None:
            domain = Domain(
                [ContinuousVariable("coef")], metas=[StringVariable("name")])
            coefs = [float(i) for i in self.model.skl_model.coef_]
            names = [attr.name for attr in self.model.domain.attributes]

            coef_table = Table.from_list(domain, list(zip(coefs, names)))
            coef_table.name = "coefficients"

            loadings = self.model.skl_model.x_loadings_
            Metas_ls = [attr.name for attr in self.model.domain.attributes]
            Metas = np.array(Metas_ls)
            metas = np.reshape(Metas,(len(Metas_ls),1))

            proposed = ['PLSC {}'.format(i + 1)
                                  for i in range(self.ncomponents)]
            meta_name = get_unique_names(proposed, 'variables')
            domain_loadings = Domain(
                [ContinuousVariable(name, compute_value=lambda _: None)
                 for name in proposed],
                metas=[StringVariable(name=meta_name)])

            loadings_table = Table(domain_loadings, loadings, metas=metas
                               )
            loadings_table.name = "x-loadings"

            scores = self.model.skl_model.x_scores_
            b=self.data.X.shape[0]
            Metas_ls_scores = [float(y) for y in self.data.Y]
            Metas_scores = np.array(Metas_ls_scores)
            metas_scores = np.reshape(Metas_scores,(len(Metas_ls_scores),1))
            scores_table = Table(domain_loadings, scores, metas=metas_scores
                               )
            scores_table.name = "x-scores"
        self.Outputs.coefficients.send(coef_table)
        self.Outputs.loadings.send(loadings_table)
        self.Outputs.scores.send(scores_table)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPLSRegression).run(Table("housing"))
