from Orange.base import Model
import numpy as np
import scipy
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.data import Table, Instance, Value
from Orange.data.util import one_hot
from orangecontrib.extension.utils.LoggingDummy import PrinLog


__all__ = ["sklmodel", "model"]

class model(Model):
    PrinLog().prinlog("initialize model class")
    def __init__(self, domain=None, original_domain=None):
        PrinLog().prinlog("__init__ func. called")
        super().__init__(self, domain=None, original_domain=None)
        PrinLog().prinlog("__super__ func. called")

    def __call__(self, data, ret=Value):
        PrinLog().prinlog("__call__ func. called")
        multitarget = len(self.domain.class_vars) > 1

        def one_hot_probs(value):
            PrinLog().prinlog("one_hot_probs func. called")
            if not multitarget:
                return one_hot(
                    value,
                    dim=len(self.domain.class_var.values)
                    if self.domain is not None else None
                )

            max_card = max(len(c.values) for c in self.domain.class_vars)
            probs = np.zeros(value.shape + (max_card,), float)
            for i in range(len(self.domain.class_vars)):
                probs[:, i, :] = one_hot(value[:, i])
            return probs

        def extend_probabilities(probs):
            PrinLog().prinlog("extend_probabilities func. called")
            """
            Since SklModels and models implementing `fit` and not `fit_storage`
            do not guarantee correct prediction dimensionality, extend
            dimensionality of probabilities when it does not match the number
            of values in the domain.
            """
            class_vars = self.domain.class_vars
            max_values = max(len(cv.values) for cv in class_vars)
            if max_values == probs.shape[-1]:
                return probs

            if not self.supports_multiclass:
                probs = probs[:, np.newaxis, :]

            probs_ext = np.zeros((len(probs), len(class_vars), max_values))
            for c, used_vals in enumerate(self.used_vals):
                for i, cv in enumerate(used_vals):
                    probs_ext[:, c, cv] = probs[:, c, i]

            if not self.supports_multiclass:
                probs_ext = probs_ext[:, 0, :]
            return probs_ext

        def fix_dim(x):
            PrinLog().prinlog("fix_dim func. called")
            return x[0] if one_d else x

        if not 0 <= ret <= 2:
            raise ValueError("invalid value of argument 'ret'")
        if ret > 0 and any(v.is_continuous for v in self.domain.class_vars):
            raise ValueError("cannot predict continuous distributions")

        # Convert 1d structures to 2d and remember doing it
        one_d = True
        if isinstance(data, Instance):
            PrinLog().prinlog("data is instance")
            data = Table.from_list(data.domain, [data])
        elif isinstance(data, (list, tuple)) \
                and not isinstance(data[0], (list, tuple)):
            PrinLog().prinlog("data is list or tuple")
            data = [data]
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            PrinLog().prinlog("data is 1d ndarray")
            data = np.atleast_2d(data)
        else:
            one_d = False

        # if sparse convert to csr_matrix
        if scipy.sparse.issparse(data):
            data = data.tocsr()

        # Call the predictor
        backmappers = None
        n_values = []

        if isinstance(data, (np.ndarray, scipy.sparse.csr.csr_matrix)):
            PrinLog().prinlog("data is ndarray")
            prediction = self.predict(data)
        elif isinstance(data, Table):
            PrinLog().prinlog("data is table")
            backmappers, n_values = self.get_backmappers(data)
            data = self.data_to_model_domain(data)
            prediction = self.predict_storage(data)
            if prediction.ndim != 1 and self.name == "plsr":
                prediction = prediction[:, 0]
                a = 1
        elif isinstance(data, (list, tuple)):
            PrinLog().prinlog("data is list or tuple")
            data = Table.from_list(self.original_domain, data)
            data = data.transform(self.domain)
            prediction = self.predict_storage(data)
        else:
            raise TypeError("Unrecognized argument (instance of '{}')"
                            .format(type(data).__name__))

        # Parse the result into value and probs
        if isinstance(prediction, tuple):
            PrinLog().prinlog("prediction is tuple")
            value, probs = prediction
        elif prediction.ndim == 1 + multitarget:
            PrinLog().prinlog("Bla1")
            value, probs = prediction, None
        elif prediction.ndim == 2 + multitarget:
            PrinLog().prinlog("Bla2")
            value, probs = None, prediction
        else:
            raise TypeError("model returned a %i-dimensional array",
                            prediction.ndim)

        # Ensure that we have what we need to return; backmapp everything
        if probs is None and (ret != Model.Value or backmappers is not None):
            probs = one_hot_probs(value)
        if probs is not None:
            probs = extend_probabilities(probs)
            probs = self.backmap_probs(probs, n_values, backmappers)
        if ret != Model.Probs:
            if value is None:
                value = np.argmax(probs, axis=-1)
                # probs are already backmapped
            else:
                value = self.backmap_value(value, probs, n_values, backmappers)

        # Return what we need to
        if ret == Model.Probs:
            return fix_dim(probs)
        if isinstance(data, Instance) and not multitarget:
            value = [Value(self.domain.class_var, value[0])]
        if ret == Model.Value:
            return fix_dim(value)
        else:  # ret == Model.ValueProbs
            return fix_dim(value), fix_dim(probs)

class sklmodel(model, metaclass=WrapperMeta):

    used_vals = None

    def __init__(self, skl_model):
        self.skl_model = skl_model


    def predict(self, X):

        PrinLog().prinlog("__ptredict__ func. called")
        value = self.skl_model.predict(X)
        if (isinstance(value[0], np.ndarray)):
            PrinLog().prinlog("value is a np.array")
            value = np.stack(value, axis=1)[0]
        # SVM has probability attribute which defines if method compute probs
        has_prob_attr = hasattr(self.skl_model, "probability")
        if (has_prob_attr and self.skl_model.probability
                or not has_prob_attr
                and hasattr(self.skl_model, "predict_proba")):
            probs = self.skl_model.predict_proba(X)
            return value, probs
        return value

    def __repr__(self):
        # Params represented as a comment because not passed into constructor
        return super().__repr__() + '  # params=' + repr(self.params)

