import numpy as np
import statsmodels.api as sm # pyright: ignore reportMissingTypeStubs
from typing import Any

from to_be_titled.types import FloatArray
from to_be_titled.solvers.solver_types import LogisticRegressionResult 
from to_be_titled.solvers import register_solver

@register_solver("statsmodels_glm")
def fit_statsmodels_glm_logistic_regression(
    x: FloatArray, 
    y: FloatArray, 
    **kwargs: Any
) -> LogisticRegressionResult:
    """Fit a logistic regression model using the statsmodels GLM framework.

    Parameters
    ----------
    x : FloatArray
        2-D design matrix of shape (n_samples, n_features).
    y : FloatArray
        2-D response vector of shape (n_samples, 1).
    **kwargs : Any
        Additional keyword arguments forwarded directly to the underlying `statsmodels` 
        GLM fit method (e.g., `maxiter`, `tol`, `method`, `cov_type`).

    Returns
    -------
    LogisticRegressionResult
        A dataclass containing the estimated coefficient vector, linear predictors,
        and fitted probabilities.

    See Also
    --------
    statsmodels.genmod.generalized_linear_model.GLM.fit : 
        Official `statsmodels` documentation for supported fitting keyword arguments and optimization options.
    """    
    model = sm.GLM(endog=y, exog=x, family=sm.families.Binomial())
    
    sm_result = model.fit(**kwargs) # pyright: ignore[reportUnknownMemberType]

    betas = np.asarray(sm_result.params).reshape(-1, 1)
    mus = np.asarray(sm_result.mu).reshape(-1, 1)
    linear_predictors = x @ betas

    return LogisticRegressionResult(
        betas=betas,
        mus=mus,
        linear_predictors=linear_predictors,
    )