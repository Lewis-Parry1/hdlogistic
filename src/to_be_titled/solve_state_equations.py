from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize, root

from to_be_titled import state_equations

from .utils import _get_hermite_roots_weights


def _se_funcs(
    kappa: float,
    ss: float,
    alpha: float,
    gh: tuple[NDArray[np.float64], NDArray[np.float64]],
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    transform: bool = True,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Construct the system of MDYPL state evolution equations.
    Returns a closure compatible with scipy.optimize.root and scipy.optimize.minimize.
    The closure evaluates the three state evolution equations
    using Gauss–Hermite quadrature and optionally performs a log-transformation
    of the optimisation variables to enforce positivity.

    Parameters
    ----------
    kappa : float
        Asymptotic ratio of the columns/rows of the design matrix (p/n).
        kappa should be in (0,1).
    ss : float
        Square root of signal strength or of corrupted signal
        strength, depending on whether `corrupted = TRUE` or not. If corrupted is
        False, then `ss` is the limit `gamma` squared of the var(X * beta). If
        corrupted is True, then `ss` is the limit `nu` squared of
        \\text{var}(X * \\hat \\beta), where \\hat{\\beta} is the maximum
        Diaconis-Ylvisaker prior penalized likelihood (MDYPL) estimator as
        computed by [mdyplFit()] with shrinkage parameter alpha.
    alpha : float
        Shrinkage parameter of the MDYPL estimator. `alpha` should be in (0,1).
    gh : tuple[NDArray[np.float64], NDArray[np.float64]]
        A tuple of 1D arrays containing Gauss Hermite quadrature nodes and weights
        used to approximate the system's expected values.
    prox_tol : float, optional
        Convergence tolerance for the Newton-Raphson estimation of
        proximal operator, by default 1e-10.
    corrupted : bool, optional
        If False, then `ss` is the square root of the signal strength.
        If True, then `ss` is the square root of the corrupted signal strength
        is the limit of the variance of the fitted values computed by mdyplFit()
        with shrinkage parameter, alpha. By default, False.
    transform : bool, optional
        If True, the returned function expects the input parameters
        (`mu`, `b`, `sigma`) to be log-transformed. The closure will
        automatically clip and exponentiate the inputs to enforce strict
        positivity and prevent numerical underflow/overflow during solver
        exploration. By default True.

    Returns
    -------
    Callable[[NDArray[np.float64]], NDArray[np.float64]]
        An objective function `g(pars)` that takes a 1D array of the three state
        evolution parameters (optionally log-transformed) and returns a 1D array
        of the evaluated residuals for the three state equations.
    """

    def g(pars: NDArray[np.float64]) -> NDArray[np.float64]:
        if transform:
            # Limit search to exp(-20) ... exp(20) to avoid numerical overflow
            pars_clipped = np.clip(pars, -20, 20)
            pars = np.exp(pars_clipped)

        if corrupted:
            mu, b, sigma = pars[0], pars[1], pars[2]
            # TODO: Check for 0 error?
            gamma = np.sqrt(ss**2 - kappa * sigma**2) / mu
            pars = np.array([mu, b, sigma])
        else:
            gamma = ss
        return state_equations._se_no_intercept(
            mu=pars[0],
            b=pars[1],
            sigma=pars[2],
            kappa=kappa,
            gamma=gamma,
            alpha=alpha,
            gh=gh,
            prox_tol=prox_tol,
        )

    return g


@dataclass
class SolverResult:
    """
    Container for the outcome of a numerical optimisation or root-finding
    procedure.

    Attributes
    ----------
    solution, NDArray[np.float64]
        Estimated state evolution parameters.

    func_value, NDArray[np.float64]
        Residual vector evaluated at ``solution``.

    message, str
        Solver termination message.

    success, bool
        Whether the solver reported successful convergence.
    """

    solution: NDArray[np.float64]
    func_value: NDArray[np.float64]
    message: str
    success: bool


def _init_solver(
    kappa: float,
    ss: float,
    alpha: float,
    start: NDArray[np.float64] | None = None,
    init_method: str = "Nelder-Mead",
    init_iter: int = 50,
    gh: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    **minimize_kwargs: Any,
) -> SolverResult:
    """
    Performs an initial unconstrained minimization to find a robust starting
    point for the main state evolution root-finding algorithm.

    This function aims to minimize the sum of squared residuals
    (the squared L2 norm) of the MDYPL state evolution equations. This prevents
    the main trust-region or Newton-based root-finders from getting stuck in local
    minima or flat gradients in extreme parameter regimes.

    Parameters
    ----------
    kappa : float
        Asymptotic ratio of the columns/rows of the design matrix (p/n).
        `kappa` should be in (0,1).
    ss : float
        Square root of signal strength or of corrupted signal
        strength, depending on whether `corrupted = TRUE` or not. If corrupted is
        False, then `ss` is the limit `gamma` squared of the var(X * beta). If
        corrupted is True, then `ss` is the limit `nu` squared of
        \\text{var}(X * \\hat \\beta), where \\hat{\\beta} is the maximum
        Diaconis-Ylvisaker prior penalized likelihood (MDYPL) estimator as
        computed by [mdyplFit()] with shrinkage parameter alpha.
    alpha : float
        Shrinkage parameter of the MDYPL estimator. `alpha` should be in (0,1).
    start : NDArray[np.float64], optional
        A 1D array of starting values for the state evolution parameters
        (`mu`, `b`, and `sigma`). These are internally log-transformed during
        minimization to enforce strict positivity and prevent underflow. If None,
        the additional starting candidates are only considered.
    init_method : str, optional
        The optimization method to be passed to `scipy.optimize.minimize`
        to minimize the squared residuals, by default 'Nelder-Mead'.
    init_iter : int, optional
        Maximum number of iterations for the initial minimization algorithm,
        by default 50.
    gh : tuple[NDArray[np.float64], NDArray[np.float64]] | None, optional
        A tuple of 1D arrays containing Gauss-Hermite quadrature nodes and weights
        used to approximate the system's expected values. By default None, in
        which case it is set to `scipy.special.roots_hermite(200)`.
    prox_tol : float, optional
        Convergence tolerance for the Newton-Raphson estimation of the
        proximal operator, by default 1e-10.
    corrupted: bool
        If False, then `ss` is the square root of the signal strength.
        If True, then `ss` is the square root of the corrupted signal strength
        is the limit of the variance of the fitted values computed by mdyplFit()
        with shrinkage parameter, alpha. By default, False.
    **minimize_kwargs : dict[str, Any], optional
        Additional keyword arguments passed directly to `scipy.optimize.minimize`.

    Returns
    -------
    SolverResult
        A dataclass containing the optimal real-space parameters (`solution`),
        the evaluated residual vector at the solution (`func_value`), the number
        of iterations performed (`iterations`), and the solver's convergence status
        (`message`, `success`).
    """
    # define multiple initial starts if user has not defined one
    candidates: list[NDArray[np.float64]] = []
    if start is not None:
        candidates.append(np.asarray(start, dtype=np.float64))

    candidates.extend(
        [
            np.array([0.5, ss, ss]),  # state equation-scaled default
            np.array([0.1, 1.0, 1.0]),  # low-mu fallback
            np.array([0.9, ss * 2, ss]),  # large mu fallback
        ]
    )

    best: tuple[float, OptimizeResult] | None = None

    gh = gh if gh is not None else _get_hermite_roots_weights(200)
    g = _se_funcs(kappa, ss, alpha, gh, prox_tol, corrupted, transform=True)

    def objective(pars_log: NDArray[np.float64]) -> float:
        r = g(pars_log)
        return float(np.dot(r, r))

    for cand in candidates:
        start_log = np.asarray(np.log(cand), dtype=np.float64)

        res = minimize(
            objective,
            start_log,
            method=init_method,
            options={"maxiter": init_iter, **minimize_kwargs.pop("options", {})},
            **minimize_kwargs,
        )  # type: ignore[call-overload]

        resid_norm = res.fun
        if best is None or resid_norm < best[0]:
            best = (resid_norm, res)

    if best is None:
        raise RuntimeError(
            f"No candidate start converged for kappa={kappa}, gamma={ss}"
        )
    else:
        res_final = best[1]
        soln = np.exp(res_final.x)
        return SolverResult(
            solution=soln,
            func_value=g(res_final.x),
            message=res_final.message,
            success=res_final.success,
        )


def _root_solver(
    kappa: float,
    ss: float,
    alpha: float,
    start: NDArray[np.float64],
    main_method: str = "hybr",
    gh: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    transform: bool = True,
    **root_kwargs: Any,
) -> SolverResult:
    """
    Executes the main root-finding algorithm to estimate the stationary point
    (`mu*`, `b*`, `sigma*`) of the MDYPL state evolution equations.

    This function uses a dedicated root-finder to estimate the true roots
    (`mu*`, `b*`, `sigma*`) where the evaluated residuals are exactly zero.
    These roots fully characterize the aggregate bias, variance, and penalization
    of the estimator in high dimensions.

    Parameters
    ----------
    kappa : float
        Asymptotic ratio of the columns/rows of the design matrix (p/n).
        `kappa` should be in (0,1).
    ss : float
        Square root of signal strength or of corrupted signal
        strength, depending on whether `corrupted = TRUE` or not. If corrupted is
        False, then `ss` is the limit `gamma` squared of the var(X * beta). If
        corrupted is True, then `ss` is the limit `nu` squared of
        \\text{var}(X * \\hat \\beta), where \\hat{\\beta} is the maximum
        Diaconis-Ylvisaker prior penalized likelihood (MDYPL) estimator as
        computed by [mdyplFit()] with shrinkage parameter alpha.
    alpha : float
        Shrinkage parameter of the MDYPL estimator. `alpha` should be in (0,1).
    start : NDArray[np.float64]
        A 1D array of starting values for `mu`, `b` and `sigma`. For extreme
        `kappa` and `gamma` regimes, it is recommended to pass a "warm"
        starting guess obtained from an initial heuristic solver.
    main_method : str, optional
        The method to be passed into `scipy.optimize.root` to find the roots of
        the state equations, by default "hybr" .
    gh : tuple[NDArray[np.float64], NDArray[np.float64]] | None, optional
        A tuple of 1D arrays containing Gauss-Hermite quadrature nodes and weights
        used to approximate the system's expected values. By default None, in which
        case it is set to `scipy.special.roots_hermite(200)`.
    prox_tol : float, optional
        Convergence tolerance for the Newton-Raphson estimation of the
        proximal operator, by default 1e-10.
    corrupted: bool, optional
        If False, then `ss` is the square root of the signal strength.
        If True, then `ss` is the square root of the corrupted signal strength
        is the limit of the variance of the fitted values computed by mdyplFit()
        with shrinkage parameter, alpha. By default, False.
    transform : bool, optional
        If True, the input parameters (`mu`, `b`, `sigma`) are internally
        log-transformed before being passed to the objective function. The closure
        will automatically clip and exponentiate the inputs to enforce strict
        positivity and prevent numerical underflow/overflow during solver
        exploration. By default True.
    **root_kwargs : dict[str, Any], optional
        Additional keyword arguments passed directly to `scipy.optimize.root`.

    Returns
    -------
    SolverResult
        A dataclass containing the optimal real-space parameters (`solution`),
        the evaluated residual vector at the solution (`func_value`), the number
        of iterations performed (`iterations`), and the solver's termination
        status and message (`message`, `success`).
    """

    gh = gh if gh is not None else _get_hermite_roots_weights(200)

    g = _se_funcs(kappa, ss, alpha, gh, prox_tol, corrupted, transform)

    start = np.log(start) if transform else start

    res = root(g, start, method=main_method, **root_kwargs)  #  type: ignore[call-overload]

    soln = np.exp(res.x) if transform else res.x  # Return in original space

    return SolverResult(
        solution=soln, func_value=g(res.x), message=res.message, success=res.success
    )


def _solve_state_equation(
    kappa: float,
    ss: float,
    alpha: float,
    start: NDArray[np.float64],
    gh: tuple[NDArray[np.float64], NDArray[np.float64]],
    root_kwargs: dict[str, Any] | None = None,
    minimize_kwargs: dict[str, Any] | None = None,
    transform: bool = True,
    corrupted: bool = False,
    init_iter: int = 50,
    init_method: str = "Nelder-Mead",
    main_method: str = "hybr",
    prox_tol: float = 1e-10,
) -> tuple[SolverResult, str]:
    """
    Solves the MDYPL state evolution equations (without intercept).

    This wrapper function execute the optional heuristic initial minimizer to
    obtain a warm start followed by a precise numerical root-finder.
    It provides a robust mechanism for finding the stationary points
    (`mu*`, `b*`, `sigma*`) of the system's state equations.

    Parameters
    ----------
    kappa : float
        Asymptotic ratio of the columns/rows of the design matrix (p/n).
        `kappa` should be in (0,1).
    ss : float
        Square root of signal strength or of corrupted signal
        strength, depending on whether `corrupted = TRUE` or not. If corrupted is
        False, then `ss` is the limit `gamma` squared of the var(X * beta). If
        corrupted is True, then `ss` is the limit `nu` squared of
        \\text{var}(X * \\hat \\beta), where \\hat{\\beta} is the maximum
        Diaconis-Ylvisaker prior penalized likelihood (MDYPL) estimator as
        computed by [mdyplFit()] with shrinkage parameter alpha.
    alpha : float
        Shrinkage parameter of the MDYPL estimator. `alpha` should be in (0,1).
    start : NDArray[np.float64]
        A 1D array of length 3 containing the initial starting values for `mu`, `b`,
        and `sigma`.
    gh : tuple[NDArray[np.float64], NDArray[np.float64]]
        A tuple of 1D arrays containing Gauss-Hermite quadrature nodes and weights
        used to approximate the system's expected values.
    root_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed directly to the main root solver
        (`scipy.optimize.root`).
    minimize_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed directly to the initial minimizer
        (`scipy.optimize.minimize`).
    corrupted: bool, optional
        If False, then `ss` is the square root of the signal strength.
        If True, then `ss` is the square root of the corrupted signal strength
        is the limit of the variance of the fitted values computed by mdyplFit()
        with shrinkage parameter, alpha. By default, False.
    transform : bool, optional
        If True, the input parameters (`mu`, `b`, `sigma`) are internally
        log-transformed during the solver exploration to enforce strict positivity
        and prevent numerical underflow/overflow. By default True.
    init_iter : int, optional
        The number of iterations to run the initial minimization algorithm. If set
        to greater than 0, the result is passed as a warm start to the main solver.
        By default 50.
    init_method : str, optional
        The optimization method to be passed to `scipy.optimize.minimize` for the
        initial warm-start phase, by default "Nelder-Mead".
    main_method : str, optional
        The root-finding method to be passed to `scipy.optimize.root` for the exact
        solution phase, by default "hybr".
    prox_tol : float, optional
        Convergence tolerance for the Newton-Raphson estimation of the
        proximal operator, by default 1e-10.

    Returns
    -------
    tuple[SolverResult, str]
        A two-element tuple containing:
        - SolverResult: dataclass with the optimal real-space parameters
          (`solution`), the evaluated residual vector (`func_value`), number
          of iterations, and the main solver's converges success/message.
        - str: a summary of the optimization chain used (e.g. which
          initial and main methods were applied).

    Raises
    ------
    TypeError
        If `start` is not an array-like sequence.
    ValueError
        If the number of parameters in `start` does not equal 3.
    """

    npar = 3

    try:
        start_len = len(start)
    except TypeError as e:
        raise TypeError("`start` must be a sequence with a length") from e
    if start_len != npar:
        raise ValueError(f"start must have length {npar}; got {start_len}")

    root_kwargs = root_kwargs or {}
    minimize_kwargs = minimize_kwargs or {}

    if init_iter > 0:
        init_result = _init_solver(
            kappa,
            ss,
            alpha,
            start,
            init_method,
            init_iter,
            gh,
            prox_tol,
            corrupted,
            **minimize_kwargs,
        )
        start = init_result.solution
        opt_chain = f"initial_method: {init_method} -> "
    else:
        opt_chain = ""

    result = _root_solver(
        kappa,
        ss,
        alpha,
        start,
        main_method,
        gh,
        prox_tol,
        corrupted,
        transform,
        **root_kwargs,
    )
    opt_chain += f"main_method: {main_method}"

    return result, opt_chain
