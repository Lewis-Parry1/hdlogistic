import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, roots_hermite


def _se0(
    mu: float,
    b: float,
    sigma: float,
    kappa: float,
    gamma: float,
    alpha: float,
    gh: NDArray[np.float64] | None = None,
    prox_tol: float = 1e-10,
) -> NDArray[np.float64]:
    """

    MDYPL state evolution functions with no intercept.

    Parameters
    ----------
    mu : float
        aggregate bias parameter.
    b : float
        parameter 'b' in state evolution functions.
    sigma : float
        square root of aggregate variance of the MDYPL estimator.
    kappa : float
        kappa asymptotic ratio of columns/rows of the design matrix. `kappa` should be
        in `(0,1)`
    gamma : float
        square root of the limit of the variance of the linear predictor.
    alpha : float
        the shrinkage parameter of the MDUPL estimator. `alpha` should be in `(0,1)`.
    gh : NDArray[np.float64], default = None
        a list with gauss-hermite quadrature nodes and weights as returned from by
        scipy.special.roots_hermite, by default is None. If None,`gh` is set to
        roots_hermite(200).
    prox_tol : float, optional
        tolerance for the computation of the proximal operator, by default 1e-10

    Returns
    -------
    NDArray[np.float64]
        Returns the estimates of the three state evolution equations with no intercept.

    References
    -------

    .. [1] Sterzinger, P., & Kosmidis, I. (2024). Diaconis-Ylvisaker prior
        penalized likelihood for p/n -> kappa in (0,1) logistic regression.
        https://arxiv.org/abs/2311.07419

    """

    xi, wi = gh if gh is not None else roots_hermite(200)

    n_nodes = len(xi)

    x_grid = np.tile(xi, n_nodes)
    y_grid = np.repeat(xi, n_nodes)

    a_frac = 0.5 * (1 + alpha)

    q1 = np.sqrt(2) * gamma * x_grid
    q2 = (q1 * mu) + np.sqrt(2) * (np.sqrt(kappa) * sigma * y_grid)

    w2 = np.tile(wi, n_nodes) * np.repeat(wi, n_nodes)

    w2p = (2 / np.pi) * w2 * expit(q1)

    p_prox = expit(prox(q2 + a_frac * b, b, prox_tol))

    prox_resid = a_frac - p_prox

    res1 = np.sum(w2p * q1 * prox_resid)
    res2 = 1 - kappa - np.sum(w2p / (1 + b * p_prox * (1 - p_prox)))
    res3 = (kappa**2 * sigma**2) - b**2 * np.sum(w2p * p_prox**2)

    return np.array([res1, res2, res3])


def prox(
    x: float | NDArray[np.float64], b: float, tol: float = 1e-10, max_iter: int = 200
) -> float | NDArray[np.float64]:
    """

    Vectorised version (in x and b) of the proximal operator.

    arg min _ u (b * log (1 + e^u) + (x-u)^2 /2)

    is minimised using Newton-Raphson.

    Parameters
    ----------
    x : float | NDArray[np.float64]
        scalar or vector of x values for proximal operator to be evaluated on
    b : float
        parameter 'b' in state evolution functions
    tol : float, optional
        convergence threshold for newton raphson step size, by default 1e-10
    max_iter : int, optional
        maximum number of Newton-Raphson updates, by default 200.

    Returns
    -------
    float | NDArray[np.float64]
        scalar (or vector) of approximation(s) of proximal operator for each x.
    """

    x_arr = np.asarray(x, dtype=float)

    u = np.zeros_like(x_arr, dtype=float)

    # First derivative when u = 0
    g0 = x_arr - b / 2

    for _ in range(max_iter):
        if np.all(np.abs(g0) < tol):
            break

        pr = expit(u)

        g0 = (x_arr - u) - b * pr
        step = g0 / (b * pr * (1 - pr) + 1)

        u = u + step

    return float(u) if x_arr.ndim == 0 else u
