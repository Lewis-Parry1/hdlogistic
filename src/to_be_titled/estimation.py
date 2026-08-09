import inspect
from typing import Any, overload

import numpy as np

from to_be_titled.solvers.registry import _SOLVERS_REGISTRY
from to_be_titled.solvers.solver_types import SolverFunction
from to_be_titled.types import (
    DiaconisYlvisakerLogisticRegressionResult,
    FloatArray,
)


def _adjust_response(y: FloatArray, alpha: float) -> FloatArray:
    """Compute the adjusted response vector under a Diaconis-Ylvisaker prior.

    Transforms the empirical binary responses into pseudo-probabilities shifted
    toward the prior distribution. This specific formulation assumes a zero prior
    mode, which evaluates the sigmoid function to 0.5.

    Parameters
    ----------
    y : FloatArray
        Original binary response vector of shape (n_samples,) or (n_samples, 1).
    alpha : float
        Shrinkage parameter in [0, 1]. Lower values enforce stronger prior
        regularization, pulling the pseudo-responses toward 0.5 (which shrinks
        coefficient estimates toward the prior mode of 0). Setting alpha = 1.0
        recovers standard unpenalized maximum likelihood estimation.

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


@overload
def fit_diaconis_ylvisaker_logistic_regression(
    x: FloatArray,
    y: FloatArray,
    intercept_index: int | None = None,
    alpha: float | None = None,
    *,
    solver: str = "fisher_scoring",
    solver_kwargs: dict[str, Any] | None = None,
) -> DiaconisYlvisakerLogisticRegressionResult: ...


@overload
def fit_diaconis_ylvisaker_logistic_regression(
    x: FloatArray,
    y: FloatArray,
    intercept_index: int | None = None,
    alpha: float | None = None,
    *,
    solver: SolverFunction,
    solver_kwargs: None = None,
) -> DiaconisYlvisakerLogisticRegressionResult: ...


def fit_diaconis_ylvisaker_logistic_regression(
    x: FloatArray,
    y: FloatArray,
    intercept_index: int | None = None,
    alpha: float | None = None,
    *,
    solver: str | SolverFunction = "fisher_scoring",
    solver_kwargs: dict[str, Any] | None = None,
) -> DiaconisYlvisakerLogisticRegressionResult:
    """Fit a logistic regression model using maximum Diaconis-Ylvisaker prior
    penalized likelihood.

    Estimates regression coefficients using a Fisher scoring method. Due to the
    properties of the Diaconis-Ylvisaker prior, simplifies to standard maximisation
    of the log-likelihood function, on an adjusted response vector.

    Parameters
    ----------
    x : FloatArray
        2-D design matrix of shape (n_samples, n_features).
    y : FloatArray
        Binary response vector of shape (n_samples,) or (n_samples, 1).
    intercept_index : int | None, default = None
        Zero-based column index corresponding to the scalar intercept term in the
        design matrix `x`. If provided, the scalar parameter estimate `theta_hat`
        is extracted from this position in the terminal coefficient vector.
    alpha : float | None, default = None
        The prior shrinkage parameter in [0, 1] in the Diaconis-Ylvisaker
        prior penalty. Default is None, in which `alpha` is set to n / (n + p)
        or equivalently, 1 / (1 + kappa). Setting `alpha` to 1.0 corresponds to
        using standard unpenalized maximum likelihood estimation.
    solver : str, SolverKind, or callable, default = SolverKind.FISHER_SCORING
        The numerical optimization solver backend to use. Accepts a registered
        solver name (e.g., `"fisher_scoring"`, `"nesterov_gradient_descent"`), a
        `SolverKind` enum, or a custom pre-configured callable (e.g., via
        `functools.partial` or lambda functions).
    solver_kwargs : dict[str, Any] | None, default = None
        Additional keyword arguments passed directly to the solver function when
        using string or enum dispatch. Must be `None` if `solver` is a pre-configured
        callable. See the respective solver function docstring for accepted options.

    Returns
    -------
    DiaconisYlvisakerLogisticRegressionResult
        A dataclass containing the estimated coefficient vector, linear predictors,
        fitted probabilities, adjusted response vector, validated design matrix,
        prior shrinkage hyperparameter (`alpha`), and the scalar intercept estimate
        (`theta_hat`).

    Raises
    ------
    ValueError
        If `alpha` is outside the closed interval [0.0, 1.0].
        If `solver` is not a recognized `SolverKind` or valid solver string.
        If `solver_kwargs` is provided alongside a pre-configured callable `solver`.
        If `x` or `y` fail structural and dimensional checks during validation.
    TypeError
        If the provided `solver_kwargs` are invalid for the chosen solver's signature.
    LinAlgError
        If the Fisher information matrix is singular or ill-conditioned and cannot
        be inverted during solver iterations.

    See Also
    --------
    to_be_titled.solvers.fit_logistic_regression_fisher_scoring :
        Fisher scoring solver backend options.
    to_be_titled.solvers.fit_logistic_regression_nesterov_accelerated_gradient_descent :
        Nesterov Accelerated Gradient Descent solver backend options.

    References
    ----------
    .. [1] Sterzinger, P., & Kosmidis, I. (2026). Diaconis-Ylvisaker prior
           penalized likelihood for p/n -> kappa in (0,1) logistic regression.
           https://arxiv.org/abs/2311.07419
    """
    # TODO: Add examples of usage with partial, lambda and string dispatch
    x_validated = _ensure_design_matrix(x)
    y_validated = _ensure_column_vector(y)

    if alpha is None:
        n, p = x_validated.shape[0], x_validated.shape[1]
        alpha = n / (n + p)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    y_adjusted = _adjust_response(y_validated, alpha=alpha)

    # Delegate to the chosen solver function
    if callable(solver):
        if solver_kwargs is not None:
            raise ValueError(
                "`solver_kwargs` cannot be provided when `solver` is a "
                "pre-configured callable. "
                "Pass arguments directly via `functools.partial` instead."
            )
        result = solver(x_validated, y_adjusted)
    else:
        if solver not in _SOLVERS_REGISTRY:
            valid_solvers = list(_SOLVERS_REGISTRY.keys())
            raise ValueError(
                f"Unknown solver '{solver}'. Available solvers: {valid_solvers}"
            )

        solver_function = _SOLVERS_REGISTRY[solver]
        kwargs = solver_kwargs or {}

        solver_function_signature = inspect.signature(solver_function)
        try:
            solver_function_signature.bind(x_validated, y_adjusted, **kwargs)
        except TypeError as e:
            raise TypeError(f"Invalid arguments for solver '{solver}': {e}") from None

        result = solver_function(x_validated, y_adjusted, **kwargs)

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
