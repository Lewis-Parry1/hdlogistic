import numpy as np

from to_be_titled.solvers.solver_types import SolverResult
from to_be_titled.types import FloatArray


def _is_valid(result: SolverResult, tol: float = 1e-4) -> bool:
    converged = bool(np.all(np.abs(result.func_value) < tol))
    return converged


def _validate_domain(params: FloatArray) -> None:
    """
    Validates `mu`, `b`, `sigma` and optional `intercept` lie within required domains.
    """
    mu, b, sigma = params[:3]

    n_param = len(params)

    if not np.all(np.isfinite(params[:n_param])):
        raise ValueError(f"All parameters must be finite. Received {params}")
    if mu <= 0:
        raise ValueError(f"`mu` must be greater than 0. Received {mu}.")
    if b <= 0:
        raise ValueError(f"`b` must be strictly positive. Received {b}.")
    if sigma <= 0:
        raise ValueError(f"`sigma` must be strictly positive. Received {sigma}.")


def _is_valid_domain(params: FloatArray) -> bool:
    """
    Returns True if `mu`, `b`, `sigma` and optional intercept, are in
    valid domain. Returns False otherwise.
    """
    try:
        _validate_domain(params)
    except ValueError:
        return False
    return True


def _validate_start_dims(params: FloatArray, has_intercept: bool) -> None:
    """
    Validates that number of variable parameters for the system of state equations
    is 3 if `has_intercept = False` and 4 otherwise.
    """
    expected_dim = 4 if has_intercept else 3
    if params.shape != (expected_dim,):
        raise ValueError(
            f"`start` must be a length-{expected_dim} vector, got shape {params.shape}"
        )


def _validate_state_equation_fixed_params(
    alpha: float, kappa: float, gamma: float
) -> None:
    if not (0 < alpha <= 1):
        raise ValueError(f"`alpha` is contained in (0,1]. Recieved `alpha` = {alpha}.")
    if not (0 < kappa < 1):
        raise ValueError(f"`kappa` is contained in (0,1). Received `kappa` = {kappa}.")
    if gamma <= 0:
        raise ValueError(
            "`gamma` is less than or equal to zero. Variance must be "
            f"strictly positive. Received `gamma` = {gamma}"
        )


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
    """Convert input to a 1-D float64 column vector and validate dimensions.

    Accepts a 1-D array or a 2-D array with a single column, and returns it
    as 1-D to be passed into `statsmodels.GLM()`.

    Parameters
    ----------
    y : FloatArray
        Input response array-like structure.

    Returns
    -------
    FloatArray
        Validated 1-D column vector of shape (n_samples, ).

    Raises
    ------
    ValueError
        If `y` cannot be represented as a 1-D vector.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError("y must be a 1-D response vector or a 2-D column vector")
    return y
