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
        self.prior_weights = model.prior_weights

        self.iterations = glm_results.fit_history.get("iteration")

        self.params = np.asarray(glm_results.params, dtype=np.float64)
        self.fitted_probs = np.asarray(self.fittedvalues, dtype=np.float64).copy()
        self.linear_predictors = cast(FloatArray, model.exog) @ self.params + model.offset 

        self.residuals = (model.y_raw - self.fitted_probs) / (self.fitted_probs * (1.0 - self.fitted_probs))  
        self.rank = len(self.params)

        self.aic = logist_aic(self.y_adj, self.fitted_probs, self.prior_weights) + 2.0 * self.rank
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

    def _compute_null_deviance(self, model: MDYPLModel) -> float: 
        """Compute deviance of model fitted with only constant feature. 
        """
        exog = cast(FloatArray, model.exog)
        intercept_idx = model.intercept_idx

        family = model.family

        if model.has_intercept: 
            intercept_col = exog[:, intercept_idx]

            null_alpha = self.alpha if model.alpha_was_fixed else None

            y_mean = float(np.average(self.y_raw, self.prior_weights))
            start_val = float(model.family.link(y_mean))

            null_model = MDYPLModel(y = self.y_raw,
                                    x = intercept_col,
                                    weights = self.prior_weights,
                                    offset = model.offset, # offset is zeros if no offset included 
                                    alpha=null_alpha,
                                    family = model.family,
                                    fit_kwargs=model.fit_kwargs
                                    )
            null_mus = null_model.fit(start_params=[start_val]).fitted_probs
            return family.deviance(self.y_adj, null_mus, self.prior_weights) # type: ignore[arg-type]

        else:
            null_mus = model.family.link.inverse(model.offset)
            return family.deviance(self.y_adj, null_mus, self.prior_weights) # type: ignore[arg-type]

