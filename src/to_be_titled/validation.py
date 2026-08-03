import numpy as np

from to_be_titled.types import FloatArray


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
