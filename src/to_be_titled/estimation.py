from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from statsmodels.genmod.families import (  # pyright: ignore[reportMissingTypeStubs]
    Binomial,
)
from statsmodels.genmod.generalized_linear_model import (  # pyright: ignore[reportMissingTypeStubs]
    GLM,
)

from to_be_titled.inference import logist_aic
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
    """Container storing validated design matrix, response data, and model metadata.

    Attributes
    ----------
    x : FloatArray
        Design matrix of shape `(n, p)`.
    y_raw : FloatArray
        Raw binary response vector of shape `(n,)`.
    weights : FloatArray
        Observation weights vector of shape `(n,)`.
    offset : FloatArray | None
        Additive offset vector of shape `(n,)`, or None.
    rank : int
        Column rank `p` of the design matrix.
    has_intercept : bool
        Whether a constant intercept column is present.
    intercept_idx : int | None
        Column index of the intercept, or None if absent.
    nobs_eff : float
        Effective sample size (sum of observation weights).
    """

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
    """Results container holding fitted parameters, predictions, and model diagnostics.

    Parameters
    ----------
    data : MDYPLData
        Internal data container used to fit the model.
    params : FloatArray
        Fitted parameter estimates of shape `(p,)`.
    linear_predictors : FloatArray
        Linear predictors `X @ beta + offset` of shape `(n,)`.
    fitted_probs : FloatArray
        Fitted probabilities of shape `(n,)`.
    deviance : float
        Model deviance at convergence.
    aic : float
        Akaike Information Criterion evaluated with adjusted response.
    converged : bool
        Whether the optimization routine converged successfully.
    iterations : int
        Number of iterations executed by the solver.
    alpha : float
        Shrinkage parameter applied to the response.
    y_adj : FloatArray
        Shrunk response vector of shape `(n,)`.
    _glm_results : Any, optional
        Underlying statsmodels `GLMResults` instance, by default None.
    """

    data: MDYPLData
    params: FloatArray
    linear_predictors: FloatArray
    fitted_probs: FloatArray
    deviance: float
    aic: float
    converged: bool
    iterations: int
    alpha: float
    y_adj: FloatArray
    _glm_results: Any = field(default=None, repr=False, compare=False)

    @property
    def x(self) -> FloatArray:
        """Design matrix of shape `(n, p)`.

        Returns
        -------
        FloatArray
            Model design matrix from the underlying data container.
        """
        return self.data.x

    @property
    def y_raw(self) -> FloatArray:
        """Raw unadjusted response vector.

        Returns
        -------
        FloatArray
            Raw response array of shape `(n,)`.
        """
        return self.data.y_raw

    @property
    def weights(self) -> FloatArray:
        """Observation weights vector.

        Returns
        -------
        FloatArray
            Weights array of shape `(n,)`.
        """
        return self.data.weights

    @property
    def offset(self) -> FloatArray | None:
        """Additive offset term.

        Returns
        -------
        FloatArray | None
            Offset array of shape `(n,)`, or None if not specified.
        """
        return self.data.offset

    @property
    def rank(self) -> int:
        """Rank of the design matrix.

        Returns
        -------
        int
            Column rank `p` of the design matrix.
        """
        return self.data.rank

    @property
    def has_intercept(self) -> bool:
        """Indicator for presence of an intercept column.

        Returns
        -------
        bool
            True if an intercept column is present, False otherwise.
        """
        return self.data.has_intercept

    @property
    def intercept_idx(self) -> int | None:
        """Column index of the intercept in the design matrix.

        Returns
        -------
        int | None
            Index of the intercept column, or None if absent.
        """
        return self.data.intercept_idx

    @property
    def nobs_eff(self) -> float:
        """Effective sample size.

        Returns
        -------
        float
            Sum of observation weights.
        """
        return self.data.nobs_eff

    @property
    def intercept(self) -> float | None:
        """Estimated intercept coefficient.

        Returns
        -------
        float | None
            Intercept parameter estimate, or None if the model has no intercept.
        """
        if self.has_intercept and self.intercept_idx is not None:
            return float(self.params[self.intercept_idx])
        return None

    @cached_property
    def leverages(self) -> FloatArray:
        """Leverage values (diagonal entries of the hat matrix).

        Returns
        -------
        FloatArray
            Array of leverages of shape `(n,)`.

        Raises
        ------
        ValueError
            If the underlying `_glm_results` instance is None.
        """
        if self._glm_results is None:
            raise ValueError("Leverages require underlying GLM results.")
        return np.asarray(
            self._glm_results.get_influence().hat_matrix_diag, dtype=np.float64
        )

    @cached_property
    def cov_params(self) -> FloatArray:
        """Estimated variance-covariance matrix of parameter estimates.

        Returns
        -------
        FloatArray
            Parameter covariance matrix of shape `(p, p)`.

        Raises
        ------
        ValueError
            If the underlying `_glm_results` instance is None.
        """
        if self._glm_results is None:
            raise ValueError("Covariance matrix requires underlying GLM results.")
        return np.asarray(self._glm_results.cov_params(), dtype=np.float64)


def prepare_mdypl_data(
    x: FloatArray,
    y: FloatArray,
    weights: FloatArray | None = None,
    offset: float | FloatArray | None = None,
) -> MDYPLData:
    """Validate, standardise, and package model inputs into an internal data container.

    Parameters
    ----------
    x : FloatArray
        Covariates/predictors convertible to a 2D design matrix of shape `(n, p)`.
    y : FloatArray
        Response vector convertible to shape `(n, 1)` or `(n,)`.
    weights : FloatArray | None, optional
        Observation-level weights of shape `(n,)`. If None, defaults to an
        array of ones, by default None.
    offset : float | FloatArray | None, optional
        Additive offset term as either a scalar (broadcast to length `n`) or
        an array of shape `(n,)`, by default None.

    Returns
    -------
    MDYPLData
        Container storing validated `x`, flattened `y`, weights, expanded
        offset, rank `p`, intercept metadata, and effective sample size.

    Raises
    ------
    ValueError
        If the validated design matrix `x` is rank-deficient.
    ValueError
        If `weights` is provided and its length does not equal sample size `n`.
    ValueError
        If `offset` is provided as an array and its length does not equal
        sample size `n`.
    """
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
    """Fit the modified Diaconis-Ylvisaker penalized logistic model.

    Parameters
    ----------
    data : MDYPLData
        Validated internal data container holding the design matrix, raw response,
        observation weights, offset, and structural metadata.
    alpha : float | None, optional
        Shrinkage parameter constrained to `[0, 1]`. If None, automatically computed
        as `nobs_eff / (nobs_eff + rank - has_intercept)`, by default None.
    tol : float, optional
        Convergence tolerance passed to the GLM solver, by default 1e-8.
    maxiter : int, optional
        Maximum number of iterations allowed for the solver, by default 100.
    method : str, optional
        Optimization algorithm for `GLM.fit` (e.g., `'IRLS'`), by default "IRLS".
    start_params : FloatArray | None, optional
        Initial parameter vector of shape `(p,)` for the optimization routine,
        by default None.

    Returns
    -------
    MDYPLResults
        Results container storing estimated coefficients, linear predictors,
        fitted probabilities, deviance, AIC, solver convergence flags, and the
        underlying `GLMResults` instance.

    Raises
    ------
    ValueError
        If `alpha` is not within the closed interval `[0, 1]`.
    """
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

    aic = float(logist_aic(y_adj, fitted_probs, data.weights) + 2.0 * data.rank)

    return MDYPLResults(
        data=data,
        params=params,
        linear_predictors=linear_predictors,
        fitted_probs=fitted_probs,
        deviance=float(glm_results.deviance),
        aic=aic,
        converged=bool(glm_results.converged),
        iterations=int(glm_results.fit_history.get("iteration", 0)),
        alpha=alpha_val,
        y_adj=y_adj,
        _glm_results=glm_results,
    )
