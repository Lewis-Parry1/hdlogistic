import numpy as np
from numpy.typing import NDArray
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from statsmodels.genmod.families.family import Binomial
from typing import Any, cast
from functools import cached_property

from to_be_titled.mdypl_fit import MDYPLModel
from to_be_titled.inference import logist_aic


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

        self.iterations = glm_results.iterations

        self.params = np.asarray(glm_results.params, dtype=np.float64)
        self.fitted_probs = np.asarray(self.fittedvalues, dtype=np.float64).copy()
        self.linear_predictors = cast(FloatArray, model.exog) @ self.params + model.offset 

        self.residuals = (model.y_raw - self.fitted_probs) / (self.fitted_probs * (1.0 - self.fitted_probs))  
        self.rank = glm_results.df_model + 1.0 # TODO: will need to be updated for singular matrices

        self.aic = logist_aic(self.y_adj, self.fitted_probs, model.weights) + 2.0 * self.rank
        self.deviance = glm_results.deviance 
        self.null_deviance = self._compute_null_deviance(model)


    def __getattr__(self, name): 
        return getattr(self._results, name)

    @property
    def intercept(self) -> float | None: 
        """The fitted intercept coeffcient, or None if the model has 
        no intercept. 
        """
        if self.intercept_idx is None:
            return None 
        return float(self.params[self.intercept_idx]) 

    @cached_property
    def leverages(self) -> FloatArray:
        """The diaganal of the hat matrix. Cached as to not need to recompute.
        """
        return np.asarray(self._results.get_hat_matrix_diag(), dtype=np.float64)

    def _compute_null_deviance(self, model: MDYPLModel) -> FloatArray: 
        """Compute deviance of model fitted with only constant feature. If
        model has no intercept then this is either expit(offset) if 
        `missing_offset` is False, or set trivially to full model deviance
        if True. 
        """
        exog = cast(FloatArray, model.exog)
        intercept_idx = model.intercept_idx

        if model.has_intercept and model.missing_offset: 
            intercept_col = exog[:, intercept_idx]
            null_model = MDYPLModel(y = model.y_raw,
                                    x = intercept_col,
                                    weights = model.weights,
                                    offset = model.offset,
                                    family = model.family,
                                    )
            null_mu = null_model.fit().fitted_probs
        elif not model.has_intercept:
            null_mu = model.family.link.inverse(model.offset)
        else:
            null_mu = self.fitted_probs

        return np.asarray(null_mu, dtype = np.float64)
