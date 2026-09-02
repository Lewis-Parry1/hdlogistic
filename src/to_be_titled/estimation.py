from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from scipy.special import expit, logit
from statsmodels.genmod.families import (  # pyright: ignore[reportMissingTypeStubs]
    Binomial,
)
from statsmodels.genmod.generalized_linear_model import (  # pyright: ignore[reportMissingTypeStubs]
    GLM,
)

from to_be_titled.inference import (
    compute_deviance, compute_deviance_residuals, compute_pearson_residuals, 
    logistic_aic, logistic_bic

)
from to_be_titled.types import (
    FloatArray,
)
from to_be_titled.utils import adjust_response, get_intercept_idx, has_constant_col
from to_be_titled.validation import (
    ensure_column_vector,
    ensure_design_matrix,
    is_full_rank,
)


@dataclass(frozen=True)
class MDYPLData:
    '''
    Struct to hold the data and metadata for fitting a logistic regression model using the MDYPL.
    '''
    x: FloatArray
    y_raw: FloatArray
    weights: FloatArray
    offset: FloatArray | None
    rank: int
    has_intercept: bool
    intercept_idx: int | None
    nobs_eff: float


@dataclass(frozen=True)
class MDYPLResults:
    '''
    Results of fitting a logistic regression model using the MDYPL. 
    '''
    data: MDYPLData
    params: FloatArray
    
    linear_predictors: FloatArray
    fitted_probs: FloatArray

    aic: float
    bic: float
    deviance: float
    null_deviance: float 

    converged: bool
    iterations: int
    alpha: float
    y_adj: FloatArray
    _glm_results: Any = field(default=None, repr=False, compare=False)

    @property
    def x(self) -> FloatArray:
        return self.data.x

    @property
    def y_raw(self) -> FloatArray:
        return self.data.y_raw

    @property
    def weights(self) -> FloatArray:
        return self.data.weights

    @property
    def offset(self) -> FloatArray | None:
        return self.data.offset

    @property
    def rank(self) -> int:
        return self.data.rank

    @property
    def has_intercept(self) -> bool:
        return self.data.has_intercept

    @property
    def intercept_idx(self) -> int | None:
        return self.data.intercept_idx

    @property
    def nobs_eff(self) -> float:
        return self.data.nobs_eff

    @property
    def intercept(self) -> float | None:
        if self.has_intercept and self.intercept_idx is not None:
            return float(self.params[self.intercept_idx])
        return None

    @cached_property
    def leverages(self) -> FloatArray:
        if self._glm_results is None:
            raise ValueError("Leverages require underlying GLM results.")
        return np.asarray(
            self._glm_results.get_influence().hat_matrix_diag, dtype=np.float64
        )

    @cached_property
    def cov_params(self) -> FloatArray:
        if self._glm_results is None:
            raise ValueError("Covariance matrix requires underlying GLM results.")
        return np.asarray(self._glm_results.cov_params(), dtype=np.float64)

    @cached_property
    def resid_deviance(self) -> FloatArray:
        return compute_deviance_residuals(self.y_adj, self.fitted_probs, 
                                          self.data.weights, eps=1e-15)
    @cached_property
    def resid_pearson(self) -> FloatArray:
        return compute_pearson_residuals(self.y_adj, self.fitted_probs,
                                         self.data.weights, eps=1e-15)


def prepare_mdypl_data(
    x: FloatArray,
    y: FloatArray,
    weights: FloatArray | None = None,
    offset: float | FloatArray | None = None,
) -> MDYPLData:
    x_val = ensure_design_matrix(x)
    y_val = ensure_column_vector(y)

    if not is_full_rank(x_val):
        raise ValueError(
            "Design matrix `x` is rank-deficient. Check for duplicate, "
            "redundant, or perfectly correlated columns, and remove them before "
            "fitting."
        )

    n, p = x_val.shape
    has_intercept = has_constant_col(x_val)
    intercept_idx = get_intercept_idx(x_val)

    if weights is None:
        weights_arr = np.ones(n, dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).ravel()
        if weights_arr.shape[0] != n:
            raise ValueError(
                f"weights length ({weights_arr.shape[0]}) does not match"
                f"sample size ({n})"
            )

    nobs_eff = float(np.sum(weights_arr))

    if offset is None:
        offset_arr = None
    else:
        offset_arr = np.asarray(offset, dtype=np.float64).ravel()
        if offset_arr.shape[0] == 1:
            offset_arr = np.repeat(offset_arr, n)
        elif offset_arr.shape[0] != n:
            raise ValueError(
                f"offset length ({offset_arr.shape[0]}) does not matchsample size ({n})"
            )

    return MDYPLData(
        x=x_val,
        y_raw=y_val.ravel(),
        weights=weights_arr,
        offset=offset_arr,
        rank=p,
        has_intercept=has_intercept,
        intercept_idx=intercept_idx,
        nobs_eff=nobs_eff,
    )


def fit_mdypl(
    data: MDYPLData,
    alpha: float | None = None,
    *,
    tol: float = 1e-8,
    maxiter: int = 100,
    method: str = "IRLS",
    start_params: FloatArray | None = None,
) -> MDYPLResults:
    
    if alpha is None:
        alpha_val = data.nobs_eff / (
            data.nobs_eff + data.rank - int(data.has_intercept)
        )
    elif not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Shrinkage parameter `alpha` must be in [0, 1], got {alpha}")
    else:
        alpha_val = alpha

    y_adj = adjust_response(data.y_raw, alpha_val)
    offset_arr = (
        np.zeros(len(data.y_raw), dtype=np.float64)
        if data.offset is None
        else data.offset
    )

    glm_model = GLM(
        endog=y_adj,
        exog=data.x,
        family=Binomial(),
        freq_weights=data.weights,
        offset=offset_arr,
    )

    fit_kwargs: dict[str, Any] = {
        "tol": tol,
        "maxiter": maxiter,
        "method": method,
    }
    if start_params is not None:
        fit_kwargs["start_params"] = start_params

    glm_results = glm_model.fit(**fit_kwargs)  # pyright: ignore[reportUnknownMemberType]

    params = np.asarray(glm_results.params, dtype=np.float64)
    linear_predictors = np.asarray(data.x @ params, dtype=np.float64)
    if data.offset is not None:
        linear_predictors = linear_predictors + data.offset
        
    fitted_probs = np.asarray(glm_results.mu, dtype=np.float64)

    # Recomputed later if hd_correction is true. 
    # This is because fitted_probs is recomputed using rescaled coefficients 
    aic = logistic_aic(y_adj, fitted_probs, data.weights, data.rank, eps=1e-15)
    bic = logistic_bic(y_adj, fitted_probs, data.weights, data.rank, eps=1e-15)

    # Recomputed later as well if hd_correction is true.
    deviance = compute_deviance(y_adj, fitted_probs, data.weights, eps=1e-15)

    # Build null model and get fitted probabilities.
    # Null model fit on same adjusted responses used by original fitted model.
    if data.has_intercept:
        y_mean = float(np.average(y_adj, weights=data.weights))
        logit_y_mean = logit(np.clip(y_mean, 1e-12, 1 - 1e-12))
        null_start_params = np.asarray([logit_y_mean], dtype=np.float64)

        intercept_col = data.x[:, data.intercept_idx].reshape(-1, 1)
        null_model = GLM(
            endog = y_adj,
            exog = intercept_col,
            family = Binomial(),
            freq_weights = data.weights,
            offset = offset_arr,
        )
        # drop the start params from the fit_kwargs to avoid passing it to the null model fit
        null_kwargs = fit_kwargs.copy()
        null_kwargs.pop("start_params", None)
        null_results = null_model.fit(start_params = null_start_params, **null_kwargs)
        null_fitted_probs = np.asarray(null_results.mu, dtype=np.float64)
    else:
        # If offset is defined then null_mus are sigmoid(offset) otherwise offset = 0
        # sigmoid(0) =  0.5
        null_fitted_probs = expit(offset_arr)

    null_deviance = compute_deviance(y_adj, null_fitted_probs, data.weights, eps=1e-15)

    return MDYPLResults(
        data=data,
        params=params,
        linear_predictors=linear_predictors,
        fitted_probs=fitted_probs,

        aic=aic,
        bic = bic,
        deviance = deviance,
        null_deviance = null_deviance,

        converged=bool(glm_results.converged),
        iterations=int(glm_results.fit_history.get("iteration", 0)),
        alpha=alpha_val,
        y_adj=y_adj,
        _glm_results=glm_results,
    )
