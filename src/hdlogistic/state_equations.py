from warnings import warn

import numpy as np
from scipy.special import expit

from hdlogistic.quadrature import get_hermite_roots_weights
from hdlogistic.types import FloatArray


def se_no_intercept(
    mu: float,
    b: float,
    sigma: float,
    kappa: float,
    gamma: float,
    alpha: float,
    hermite_roots_weights: tuple[FloatArray, FloatArray] | None = None,
    prox_tol: float = 1e-10,
) -> FloatArray:
    r"""

    MDYPL state evolution functions with no intercept.

    Parameters
    ----------
    mu : float
            Aggregate bias parameter bounded between (0,1).
    b : float
        State evolution system parameter, `b` > 0.
    sigma : float
        Square root of the aggregate variance of the MDYPL estimator,
        `sigma` > 0.
    kappa : float
        Asymptotic ratio of columns/rows of the design matrix. `kappa` should be
        in `(0,1)`.
    gamma : float
        Square root of the limit of the variance of the linear predictor.
    alpha : float
        The shrinkage parameter of the MDYPL estimator. `alpha` should be in `(0,1]`.
    hermite_roots_weights : tuple[FloatArray, FloatArray] | None
        A list with gauss-hermite quadrature nodes and weights as returned from by
        scipy.special.roots_hermite, by default is None. If None,`gh` is set to
        roots_hermite(200). By default, None.
    prox_tol : float, optional
        Convergence tolerance for the computation of the proximal operator,
        by default 1e-10.

    Returns
    -------
    FloatArray
        A 1D array containing the three evaluated residuals of the MDYPL state
        evolution equations without intercept.

    References
    -------

    .. [1] Sterzinger, P., & Kosmidis, I. (2026). Diaconis-Ylvisaker prior
        penalized likelihood for p/n -> kappa in (0,1) logistic regression.
        https://arxiv.org/abs/2311.07419

    """

    xi, wi = (
        hermite_roots_weights
        if hermite_roots_weights is not None
        else get_hermite_roots_weights(200)
    )

    n_nodes = len(xi)

    # Compute 2D grid of Hermite polynomial nodes and 2D grid of weights
    x_grid = np.tile(xi, n_nodes)
    y_grid = np.repeat(xi, n_nodes)
    w_grid = np.tile(wi, n_nodes) * np.repeat(wi, n_nodes)

    # Precompute needed quantities to evaluate input to expectation in state equations
    a_frac = 0.5 * (1 + alpha)
    q1 = np.sqrt(2) * gamma * x_grid
    q2 = (q1 * mu) + np.sqrt(2) * (np.sqrt(kappa) * sigma * y_grid)

    expit_q1 = np.asarray(expit(q1), dtype=np.float64)
    # Precompute needed quantity to approximate expectation in state evolution
    # equations
    w_pi2_q1 = (2 / np.pi) * w_grid * expit_q1

    # Evaluate proximal operator for x = q2 + a_frac * b and b = b as inputs
    prox_input = _proximal_operator(q2 + a_frac * b, b, prox_tol)

    # Precompute quantities related to proximal operator needed for
    # evaluation of expectations in state evolution equations
    prox_expit = expit(prox_input)
    prox_resid = a_frac - prox_expit

    # Evaluate the three state equations given parameters
    res1 = np.sum(w_pi2_q1 * q1 * prox_resid)
    res2 = 1 - kappa - np.sum(w_pi2_q1 / (1 + b * prox_expit * (1 - prox_expit)))
    res3 = (kappa**2 * sigma**2) - b**2 * np.sum(w_pi2_q1 * prox_resid**2)

    return np.array([res1, res2, res3])


def se_with_intercept(
    mu: float,
    b: float,
    sigma: float,
    iota: float,
    kappa: float,
    gamma: float,
    alpha: float,
    theta: float,
    hermite_roots_weights: tuple[FloatArray, FloatArray] | None = None,
    prox_tol: float = 1e-10,
) -> FloatArray:
    r"""
    Evaluates the system of 4 MDYPL state evolution equations with an intercept.

    Parameters
    ----------
    mu : float
        Aggregate bias parameter bounded between (0,1).
    b : float
        State evolution system parameter, `b` > 0.
    sigma : float
        Square root of the aggregate variance of the MDYPL estimator,
        `sigma` > 0.
    iota: float
        Asymptotic limit of the MDYPL sample estimated intercept \hat{\theta}_0
        as n,p \to \infty with p/n \to \kappa.
    kappa : float
        Asymptotic ratio of columns/rows of the design matrix. `kappa` should be
        in `(0,1)`.
    gamma : float
        Square root of the limit of the variance of the linear predictor.
    alpha : float
        The shrinkage parameter of the MDYPL estimator. `alpha` should be in `(0,1]`.
    theta : float
        The true population intercept \theta_0 of the logistic regresion model.
    hermite_roots_weights : tuple[FloatArray, FloatArray] | None
        A tuple with gauss-hermite quadrature nodes and weights as returned from by
        scipy.special.roots_hermite, by default is None. If None,`gh` is set to
        roots_hermite(200). By default, None.
    prox_tol : float, optional
        Convergence tolerance for the computation of the proximal operator,
        by default 1e-10

    Returns
    -------
    FloatArray
        A 1D array of length 4 containing the evaluated residuals of the
        four state evolution equations.

    References
    -------

    .. [1] Sterzinger, P., & Kosmidis, I. (2026). Diaconis-Ylvisaker prior
        penalized likelihood for p/n -> kappa in (0,1) logistic regression.
        https://arxiv.org/abs/2311.07419

    """

    xi, wi = (
        hermite_roots_weights
        if hermite_roots_weights is not None
        else get_hermite_roots_weights(200)
    )

    n_nodes = len(xi)

    # Compute 2D grid of Hermite polynomial nodes and 2D grid of weights
    x_grid = np.tile(xi, n_nodes)
    y_grid = np.repeat(xi, n_nodes)
    w_grid = np.tile(wi, n_nodes) * np.repeat(wi, n_nodes)

    # Precompute needed quantities to evaluate input to expectation in state equations
    # q1 and q2 are transformed values which allow us to perform GH quadrature
    a_frac = 0.5 * (1 + alpha)

    # q1 and q2 specify the two transformations of variables required to perform
    # Gauss-Hermite quadrature
    q1_no_int = np.sqrt(2) * gamma * x_grid
    q1 = q1_no_int + theta

    expit_q1_pos = np.asarray(expit(q1), dtype=np.float64)
    expit_q1_neg = np.asarray(expit(-q1), dtype=np.float64)

    q2 = (q1_no_int * mu) + (np.sqrt(2 * kappa) * sigma * y_grid) + iota

    # Compute Q+ and Q- values
    prox_input_pos = a_frac * b + q2
    prox_input_neg = a_frac * b - q2

    prox_pos = _proximal_operator(prox_input_pos, b, prox_tol)
    prox_neg = _proximal_operator(prox_input_neg, b, prox_tol)

    expit_prox_pos = expit(prox_pos)
    expit_prox_neg = expit(prox_neg)

    q_pos = a_frac - expit_prox_pos
    q_neg = a_frac - expit_prox_neg

    # Evaluate the four state equations
    w_pi = w_grid / np.pi

    res1 = np.sum(w_pi * ((expit_q1_pos * q1 * q_pos) - (expit_q1_neg * q1 * q_neg)))

    denom_pos = 1 + b * expit_prox_pos * (1 - expit_prox_pos)
    denom_neg = 1 + b * expit_prox_neg * (1 - expit_prox_neg)
    res2 = (
        1
        - kappa
        - np.sum(w_pi * ((expit_q1_pos / denom_pos) + (expit_q1_neg / denom_neg)))
    )
    res3 = (kappa**2 * sigma**2) - b**2 * np.sum(
        w_pi * ((expit_q1_pos * q_pos**2) + (expit_q1_neg * q_neg**2))
    )

    res4 = np.sum(w_pi * (expit_q1_pos * q_pos - expit_q1_neg * q_neg))

    return np.array([res1, res2, res3, res4])


def _proximal_operator(
    x: float | FloatArray,
    b: float,
    rtol: float = 1e-10,
    max_iter: int = 100_000,
) -> float | FloatArray:
    """
    The function finds the scalar u which minimises (b * log (1 + e^u) + (x-u)^2 /2).
    This is known as the proximal operator [1]. The function is vectorised to take
    a vector of nodes (x) and scalar b, and return corresponding minimums. The
    Newton-Raphson algorithm is utilised in order to approximate the
    minimum of the function.

    Parameters
    ----------
    x : float | FloatArray
        Scalar or vector of x values for proximal operator to be evaluated on.
    b : float
        Parameter 'b' in state evolution functions.
    rtol : float, optional
        Convergence threshold for newton raphson step size, by default 1e-10.
    max_iter : int, optional
        Maximum number of Newton-Raphson updates, by default 10000.

    Returns
    -------
    float | FloatArray
        Scalar (or vector) of approximation(s) of proximal operator for each x.

    References
    -------

    .. [1] Sterzinger, P., & Kosmidis, I. (2024). Diaconis-Ylvisaker prior
        penalized likelihood for p/n -> kappa in (0,1) logistic regression.
        https://arxiv.org/abs/2311.07419
    .. [2] Naumann, U. (2020). Newton's Method I.
        https://www.stce.rwth-aachen.de/files/elearning/Newton_I.pdf
    """

    x_arr = np.asarray(x, dtype=float)
    u = np.zeros_like(x_arr, dtype=float)

    for _ in range(max_iter):
        expit_u = expit(u)
        g0 = (x_arr - u) - b * expit_u

        # Use adaptive tolerance; when magnitude of x gets meaningfully
        # large, convergence tolerance becomes less strict
        if np.all(np.abs(g0) < rtol * (1 + np.abs(x_arr))):
            break

        step = g0 / (b * expit_u * (1 - expit_u) + 1)

        u = u + step
    else:
        warn(
            f"Proximal operator did not converge within {max_iter} iterations.",
            RuntimeWarning,
        )

    return float(u) if x_arr.ndim == 0 else u
