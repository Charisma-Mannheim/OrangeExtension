from orangecontrib.extension.utils.baseExtension import model, sklmodel
from Orange.base import Learner, SklLearner

__all__ = ["LearnerRegression", "ModelRegression",
           "SklModelRegression", "SklLearnerRegression"]


class LearnerRegression(Learner):
    learner_adequacy_err_msg = "Continuous class variable expected."

    def check_learner_adequacy(self, domain):
        return domain.has_continuous_class


class ModelRegression(model):
    pass


class SklModelRegression(sklmodel, ModelRegression):
    pass


class SklLearnerRegression(SklLearner, LearnerRegression):
    __returns__ = SklModelRegression
