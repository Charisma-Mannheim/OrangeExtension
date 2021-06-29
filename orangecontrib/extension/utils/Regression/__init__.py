# Pull members from modules to Orange.regression namespace
# pylint: disable=wildcard-import

from .base_regression import (ModelRegression as Model,
                              LearnerRegression as Learner,
                              SklModelRegression as SklModel,
                              SklLearnerRegression as SklLearner)
from .linearExtension import *

