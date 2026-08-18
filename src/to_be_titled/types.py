import numpy as np
from numpy.typing import NDArray
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from typing import Any, cast
from functools import cached_property

from to_be_titled.mdypl_fit import MDYPLModel


type FloatArray = NDArray[np.float64]

class MDYPLResults:
    def __init__(self, model: MDYPLModel, glm_results: GLMResultsWrapper | Any):
        self.model = model
        self._results = glm_results

        self.y_raw = model.y_raw
        self.y_adj = model.y_adj
        self.alpha = model.alpha 
        self.has_intercept = model.has_intercept
        self.intercept_idx = model.intercept_idx

        # cache commonly accessed values from glm_results 
        self.params = np.asarray(glm_results.params, dtype=np.float64).copy() 
        self.nobs = glm_results.nobs
        self.fitted_probs = np.asarray(glm_results.fittedvalues, dtype = np.float64).copy()
        self.linear_predictors = cast(FloatArray, model.exog) @ self.params + (model.offset)
    

    def __getattr__(self, name): 
        return getattr(self._results, name)

    @property
    def intercept(self) -> float | None: 
        """The fitted intercept coeffcient, or None if the model has 
        no intercept. 

        Returns
        -------
        float | None
            Intercept of the fitted MDYPL model. 
        """
        if self.intercept_idx is None:
            return None 
        return float(self.params[self.intercept_idx]) 

    @cached_property
    def leverages(self) -> FloatArray:
        """The diaganal of the hat matrix. 
        """
        return np.asarray(self._results.get_hat_matrix_diag(), dtype=np.float64)

