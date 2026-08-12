from typing import Any

import numpy as np
import statsmodels.api as sm  # pyright: ignore reportMissingTypeStubs

from to_be_titled.types import FloatArray, LogisticRegressionResult


def fit_logistic_regression(
    x: FloatArray,
    y: FloatArray,
    *,
    var_weights: FloatArray | None = None,
    start_params: FloatArray | None = None,
    maxiter: int | None = None,
    tol: float | None = None,
    method: str | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> LogisticRegressionResult:
    """Fit a logistic regression model using the statsmodels GLM framework.

    Parameters
    ----------
    x : FloatArray
        2-D design matrix of shape (n_samples, n_features).
    y : FloatArray
        2-D response vector of shape (n_samples, 1).
    var_weights : FloatArray | None, default = None
        1-D or 2-D array of variance weights assigned to each observation.
    start_params : FloatArray | None, default = None
        Initial values for the regression coefficients.
    maxiter : int | None, default = None
        Maximum number of optimization iterations.
    tol : float | None, default = None
        Convergence tolerance for optimization.
    method : str | None, default = None
        Optimization solver method (e.g., `'IRLS'`).
    fit_kwargs : dict[str, Any] | None, default = None
        Additional keyword arguments forwarded directly to the underlying `statsmodels`
        GLM fit method (e.g., `cov_type`, `scale`).

    Returns
    -------
    LogisticRegressionResult
        A dataclass containing the estimated coefficient vector, linear predictors,
        and fitted probabilities.

    See Also
    --------
    statsmodels.genmod.generalized_linear_model.GLM.fit :
        Official `statsmodels` documentation for supported fitting keyword arguments
        and optimization options.
    """'
    
    if var_weights is None:
        weights_array = np.ones((x.shape[0], 1), dtype=np.float64)
    else:
        weights_array = np.asarray(var_weights, dtype=np.float64).reshape(-1, 1)

    model = sm.GLM(
        endog=y,
        exog=x,
        family=sm.families.Binomial(),
        var_weights=weights_array,
    )

    kwargs: dict[str, Any] = {}
    if start_params is not None:
        kwargs["start_params"] = start_params
    if maxiter is not None:
        kwargs["maxiter"] = maxiter
    if tol is not None:
        kwargs["tol"] = tol
    if method is not None:
        kwargs["method"] = method

    if fit_kwargs:
        kwargs.update(fit_kwargs)

    sm_result = model.fit(**kwargs)  # pyright: ignore[reportUnknownMemberType]

    betas = np.asarray(sm_result.params).reshape(-1, 1)
    mus = np.asarray(sm_result.mu).reshape(-1, 1)
    linear_predictors = x @ betas

    # Compute leverages directly using the resolved weights array (no branching)
    working_weights = weights_array * mus * (1.0 - mus)
    normalized_cov = np.asarray(sm_result.normalized_cov_params)
    leverages = working_weights * np.sum((x @ normalized_cov) * x, axis=1, keepdims=True)

    return LogisticRegressionResult(
        betas=betas,
        mus=mus,
        linear_predictors=linear_predictors,
        leverages=leverages,
    )
