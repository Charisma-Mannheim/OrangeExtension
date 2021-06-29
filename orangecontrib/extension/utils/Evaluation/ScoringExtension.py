import numpy as np
from orangecontrib.extension.utils.LoggingDummy import PrinLog
from Orange.evaluation.scoring import RegressionScore




__all__ = ["SE", "RE"]

class SE(RegressionScore):
    long_name = "Standard error"

    #PrinLog().prinlog("Initialize SE class")
    #prinlog(" Initialisiere SE... ")
    #logger.info("SE wird initialisiert")
    def compute_score(self, results):
      #  PrinLog().prinlog("compute_score (SE) func. called")
        n = results.actual.size
        y_pred = results.predicted
        y_actual = results.actual
      #  PrinLog().prinlog(str(y_actual))
      #  PrinLog().prinlog(str(y_pred))
        #mean of residuals
        mean = np.sum(y_actual - y_pred)/n
        # standard error
        se = np.array([np.sqrt(np.sum(((y_actual - y_pred)-mean) ** 2)/(n-1))])
      #  PrinLog().prinlog(str(se))
       # PrinLog().prinlog("SE computed")
        return se

class RE(RegressionScore):
    long_name = "Relative Prediction Error"
   # PrinLog().prinlog("Initialise RE class")
    def compute_score(self, results):
       # PrinLog().prinlog("compute_score (RE) func. called")
        y_pred = results.predicted
        y_actual = results.actual
        #PrinLog().prinlog(str(y_actual))
        #PrinLog().prinlog(str(y_pred))
        re = np.empty(1)
        re[0] = np.sqrt(np.sum((y_actual - y_pred)**2)/np.sum(y_actual**2))*100
        #PrinLog().prinlog(str(re))
        #PrinLog().prinlog("RE computed")
        return re