from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from to_be_titled.solvers.logistic_regression_fisher_solver import (
    fit_logistic_regression_fisher_scoring,
)
from to_be_titled.types import (
    DiaconisYlvisakerLogisticRegressionResult,
    FloatArray,
    LogisticRegressionResult,
)

SupportedSolver = Literal["fisher_scoring"]

SOLVERS_REGISTRY: dict[SupportedSolver, Callable[..., LogisticRegressionResult]] = {
    "fisher_scoring": fit_logistic_regression_fisher_scoring,
}


def _adjust_response(y: FloatArray, alpha: float) -> FloatArray:
    """Compute the adjusted response vector under a Diaconis-Ylvisaker prior.

    Transforms the empirical binary responses into pseudo-probabilities shifted
    toward the prior distribution. This specific formulation assumes a zero prior
    mode, which evaluates the inverse link function to 0.5.

    Parameters
    ----------
    y : FloatArray
        Original binary response vector of shape (n_samples,) or (n_samples, 1).
    alpha : float
        Prior shrinkage hyperparameter in [0, 1]. Lower values enforce stronger
        shrinkage toward the prior weight of 0; alpha = 1.0 recovers the
        original response vector.

    Returns
    -------
    FloatArray
        Adjusted response vector of the same shape as y, with values continuous
        on the interval [0, 1].
    """
    return alpha * y + (1 - alpha) / 2


def _ensure_design_matrix(x: FloatArray) -> FloatArray:
    """Convert input to a 2-D float64 design matrix and validate dimensions.

    Parameters
    ----------
    x : FloatArray
        Input feature array-like structure.

    Returns
    -------
    FloatArray
        Validated 2-D design matrix of shape (n_samples, n_features).

    Raises
    ------
    ValueError
        If `x` cannot be reshaped into or validated as a 2-D matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError("x must be a 2-D design matrix")
    return x


def _ensure_column_vector(y: FloatArray) -> FloatArray:
    """Convert input to a 2-D float64 column vector and validate dimensions.

    Parameters
    ----------
    y : FloatArray
        Input response array-like structure.

    Returns
    -------
    FloatArray
        Validated 2-D column vector of shape (n_samples, 1).

    Raises
    ------
    ValueError
        If `y` cannot be represented as a 1-D vector or a 2-D column vector.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim == 2 and y.shape[1] != 1:
        raise ValueError("y must be a 1-D response vector or a 2-D column vector")
    elif y.ndim > 2:
        raise ValueError("y must be a 1-D response vector or a 2-D column vector")
    return y


def fit_diaconis_ylvisaker_logistic_regression(
    x: FloatArray,
    y: FloatArray,
    alpha: float = 0.5,
    intercept_index: int | None = None,
    solver: str = "fisher_scoring",
    solver_config: dict[str, Any] = {},
) -> DiaconisYlvisakerLogisticRegressionResult:
    """Fit a logistic regression model using maximum Diaconis-Ylvisaker prior
        penalized likelihood.

        Estimates regression coefficients using a Fisher scoring method. Due to the
        properties of the Diaconis-Ylvisaker prior, simplifies to standard maximisation
        of the log-likelihood function, on an adjusted response vector.

        Parameters
        ----------
        x : FloatArray
            Design matrix of shape (n_samples, n_features).
        y : FloatArray
            Binary response vector of shape (n_samples,) or (n_samples, 1).
        alpha : float, default=0.5
            Prior shrinkage hyperparameter in [0, 1]. Controls the variance of
            the prior distribution. As alpha approaches 0, estimates shrink toward
            the prior mode, as alpha approaches 1, the maximum likelihood
            estimate is recovered.
        intercept_index : int | None, default=None
            Zero-based column index of the intercept in the design matrix `x`.
            If provided, the corresponding estimated coefficient is stored as `theta_hat`
            in the result for downstream state evolution calculations. If None, the
            model is treated as having no intercept.
        solver : str, default="fisher_scoring"
            The optimization solver backend to use.
        solver_config : dict[str, Any]
                Configuration options dictionary passed to the solver. Supported keys
                depend on the chosen solver:

                - For `"fisher_scoring"`:
                    * `"max_iterations"` (int, default=25): Maximum Fisher scoring
                    iterations.
                    * `"tolerance"` (float, default=1e-6): Convergence threshold.
                    * `"epsilon"` (float, default=1e-8): Numerical stability threshold.

        Returns
        -------
    -------
        DiaconisYlvisakerLogisticRegressionResult
            A dataclass containing the estimated coefficient vector, linear predictors,
            fitted probabilities, adjusted response, validated design matrix, prior
            shrinkage hyperparameter, and the estimated scalar intercept parameter
            (`theta`)

        Raises
        ------
        ValueError
            If `alpha` is not in the closed interval [0.0, 1.0].
            If `solver` is not recognized.
            If `x` or `y` fail structural checks during validation.
        LinAlgError
            If the Fisher information matrix is singular during solver operations.

        References
        ----------
        .. [1] Sterzinger, P., & Kosmidis, I. (2024). Diaconis-Ylvisaker prior
               penalized likelihood for p/n -> kappa in (0,1) logistic regression.
               https://arxiv.org/abs/2311.07419
    """

    if solver not in SOLVERS_REGISTRY:
        raise ValueError(
            f"Unknown solver '{solver}'. "
            f"Available solvers: {list(SOLVERS_REGISTRY.keys())}"
        )
    solver_function = SOLVERS_REGISTRY[solver]

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    x_validated = _ensure_design_matrix(x)
    y_validated = _ensure_column_vector(y)
    y_adjusted = _adjust_response(y_validated, alpha=alpha)

    # Delegate to the chosen solver function
    result = solver_function(x_validated, y_adjusted, config=solver_config)

    theta_hat = (
        float(result.betas[intercept_index, 0]) if intercept_index is not None else None
    )

    return DiaconisYlvisakerLogisticRegressionResult(
        betas=result.betas,
        linear_predictors=result.linear_predictors,
        mus=result.mus,
        y_adjusted=y_adjusted,
        x_validated=x_validated,
        alpha=alpha,
        theta_hat=theta_hat,
    )
