
from orangecontrib.extension.utils.Regression.linearExtension import PLSRLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Output
from Orange.widgets import settings, gui
from orangecontrib.extension.utils.LoggingDummy import PrinLog

from Orange.data import Table, Domain, ContinuousVariable, StringVariable





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

    #: Types
    ncomponents = settings.Setting(3)
    autosend = settings.Setting(True)
    MAX_COMPONENTS = 100

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, "Components Selection")

        self.components_spin = gui.spin(
            box, self, "ncomponents", 1, self.MAX_COMPONENTS,
            callback=self._update_selection_component_spin,
            keyboardTracking=False
        )
        #self.components_spin.setSpecialValueText("All")


    def handleNewSignals(self):
        self.apply()

    def _update_selection_component_spin(self):
        # cut changed by "ncomponents" spin.
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
        if self.model is not None:
            domain = Domain(
                [ContinuousVariable("coef")], metas=[StringVariable("name")])
            coefs = [float(i) for i in self.model.skl_model.coef_]
            names = [attr.name for attr in self.model.domain.attributes]

            coef_table = Table.from_list(domain, list(zip(coefs, names)))
            coef_table.name = "coefficients"
        self.Outputs.coefficients.send(coef_table)



if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPLSRegression).run(Table("housing"))
