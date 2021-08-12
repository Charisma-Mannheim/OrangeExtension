
from sklearn.cross_decomposition import PLSRegression as PLSR
from orangecontrib.extension.utils.baseExtension import sklmodel
from Orange.regression.linear import SklLearner, _FeatureScorerMixin
from orangecontrib.extension.utils.LoggingDummy import PrinLog

__all__ = ["PLSRLearner"]


class PLSRLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = PLSR
    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y, W=None):
         model = super().fit(X, Y, W)
         return PLSModel(model.skl_model)


class PLSModel(sklmodel):
    @property
    def intercept(self):
        return 0

    @property
    def coefficients(self):
        return self.skl_model.coef_

    def __str__(self):
        return 'PLSModel {}'.format(self.skl_model)

