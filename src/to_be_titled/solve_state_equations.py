from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize, root

from to_be_titled import state_equations

from .utils import _get_hermite_roots_weights

@dataclass
class StateParameters():
    mu : float
    b : float
    sigma : float 
    iota : float | None

    def to_array(self) -> NDArray[np.float64]: 
        if self.iota is None: 
            return np.array([self.mu, self.b, self.sigma])
        else: 
            return np.array([self.mu, self.b, self.sigma, self.iota])

def _se_funcs(
    kappa: float,
    signal_strength: float,
    alpha: float,
    gh: tuple[NDArray[np.float64], NDArray[np.float64]],
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    transform: bool = True,
    intercept: float | None = None,
    iota: float | None = None,
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
    signal_strength : float
        Square root of signal strength or of corrupted signal
        strength, depending on whether `corrupted = TRUE` or not. If corrupted is
        False, then `signal_strength` is the limit `gamma` squared of the var(X * beta). If
        corrupted is True, then `signal_strength` is the limit `nu` squared of
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
        If False, then `signal_strength` is the square root of the signal strength.
        If True, then `signal_strength` is the square root of the corrupted signal strength
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
        pars = np.asarray(pars, dtype=np.float64)
        # if using corrupted signal strength (estimated)
        if corrupted:
            # if no iota term
            if iota is None:
                # 3 param; mu , b , sigma, no intercept in model
                # clip parameters before exponentiating to avoid overflow
                pars_t = np.exp(np.clip(pars, -20, 20)) if transform else pars
                mu, b, sigma = pars_t[0], pars_t[1], pars_t[2]
                with np.errstate(invalid="ignore"):
                    # estimate gamma using corrupted ss
                    gamma = np.sqrt(signal_strength**2 - kappa * sigma**2) / mu
                if np.isnan(gamma):
                    # if we get get divide by zero error
                    return np.full(3, np.nan)
                return state_equations._se_no_intercept(
                    mu=mu,
                    b=b,
                    sigma=sigma,
                    kappa=kappa,
                    gamma=gamma,
                    alpha=alpha,
                    gh=gh,
                    prox_tol=prox_tol,
                )
            # if iota term
            else:
                # 4 param; mu, b, sigma (transformed) and intercept (untransformed)
                # iota is fixed (estimated intercept from model), solve for
                # true intercept
                if transform:
                    pars_t = pars.copy()
                    pars_t[:3] = np.exp(np.clip(pars[:3], -20, 20))
                else:
                    pars_t = pars
                mu, b, sigma, intercept_free = (
                    pars_t[0],
                    pars_t[1],
                    pars_t[2],
                    pars_t[3],
                )
                with np.errstate(invalid="ignore"):
                    gamma = np.sqrt(signal_strength**2 - kappa * sigma**2) / mu
                if np.isnan(gamma):
                    return np.full(4, np.nan)
                return state_equations._se_with_intercept(
                    mu=mu,
                    b=b,
                    sigma=sigma,
                    iota=iota,
                    kappa=kappa,
                    gamma=gamma,
                    alpha=alpha,
                    intercept=intercept_free,
                    gh=gh,
                    prox_tol=prox_tol,
                )
        else:
            if intercept is None:
                # 3 param model; mu, b , sigma
                pars_t = np.exp(np.clip(pars, -20, 20)) if transform else pars
                return state_equations._se_no_intercept(
                    mu=pars_t[0],
                    b=pars_t[1],
                    sigma=pars_t[2],
                    kappa=kappa,
                    gamma=signal_strength,
                    alpha=alpha,
                    gh=gh,
                    prox_tol=prox_tol,
                )
            # 4 param model; mu, b, sigma (transformed) and iota (untransformed)
            # true intercept is known (fixed), solve for iota
            else:
                if transform:
                    pars_t = pars.copy()
                    pars_t[:3] = np.exp(np.clip(pars[:3], -20, 20))
                else:
                    pars_t = pars
                mu, b, sigma, iota_free = pars_t[0], pars_t[1], pars_t[2], pars_t[3]
                return state_equations._se_with_intercept(
                    mu=mu,
                    b=b,
                    sigma=sigma,
                    iota=iota_free,
                    kappa=kappa,
                    gamma=signal_strength,
                    alpha=alpha,
                    intercept=intercept,
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
    solution, StateParameters
        Estimated state evolution parameters.

    func_value, NDArray[np.float64]
        Residual vector evaluated at ``solution``.

    message, str
        Solver termination message.

    success, bool
        Whether the solver reported successful convergence.
    """

    solution: StateParameters
    func_value: NDArray[np.float64]
    message: str
    success: bool

def _validate_start(start: NDArray[np.float64], has_intercept: bool) -> None: 
    '''
    Validate user supplied starting values.
    '''
    # validate start dimensions 
    expected_dim = 4 if has_intercept else 3
    if start.shape != (expected_dim, ):
        raise ValueError(f'`start` must be a length-{expected_dim} vector, got shape {start.shape}') 

    mu, b, sigma = start[:3]

    if not (0 < mu < 1): 
        raise ValueError(f'`mu` must lie in (0,1). Received {mu}.')
    if b <= 0: 
        raise ValueError(f'`b` must be strictly positive; received {b}.')
    if sigma <= 0: 
        raise ValueError(f'`sigma` must be strictly positive; received {sigma}.')
    

def _init_solver(
    kappa: float,
    signal_strength: float,
    alpha: float,
    start: NDArray[np.float64] | None = None,
    init_method: str = "Nelder-Mead",
    init_iter: int = 50,
    gh: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    intercept: float | None = None,
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
    signal_strength : float
        Square root of signal strength or of corrupted signal
        strength, depending on whether `corrupted = TRUE` or not. If corrupted is
        False, then `signal_strength` is the limit `gamma` squared of the var(X * beta). If
        corrupted is True, then `signal_strength` is the limit `nu` squared of
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
        If False, then `signal_strength` is the square root of the signal strength.
        If True, then `signal_strength` is the square root of the corrupted signal strength
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
    has_intercept = intercept is not None

    # define multiple initial starts if user has not defined one
    candidates: list[NDArray[np.float64]] = []

    if start is not None:
        start = np.asarray(start, dtype= np.float64)

        _validate_start(start, has_intercept)

        candidates.append(start)

    if has_intercept:
        candidates.extend(
            [
                np.array([0.5, signal_strength, signal_strength, 0.0]),  # state equation-scaled default
                np.array([0.1, 1.0, 1.0, 0.0]),  # low-mu fallback
                np.array([0.9, signal_strength * 2, signal_strength, 0.0]),  # large mu fallback
            ]
        )
    else:
        candidates.extend(
            [
                np.array([0.5, signal_strength, signal_strength]),  # state equation-scaled default
                np.array([0.1, 1.0, 1.0]),  # low-mu fallback
                np.array([0.9, signal_strength * 2, signal_strength]),  # large mu fallback
            ]
        )

    best: tuple[float, OptimizeResult] | None = None

    gh = gh if gh is not None else _get_hermite_roots_weights(200)

    if corrupted:
        # fitted model returns estimated intercept (iota)
        g = _se_funcs(
            kappa, signal_strength, alpha, gh, prox_tol, corrupted, transform=True, iota=intercept
        )
    else:
        # true intercept known, trying to estimate iota
        g = _se_funcs(
            kappa,
            signal_strength,
            alpha,
            gh,
            prox_tol,
            corrupted,
            transform=True,
            intercept=intercept,
        )

    def objective(pars_log: NDArray[np.float64]) -> float:
        r = g(pars_log)
        return float(np.dot(r, r))

    options_override = minimize_kwargs.pop("options", {})

    for cand in candidates:
        cand = np.asarray(cand, dtype=np.float64)

        start_vec = (
            np.concatenate([np.log(cand[:3]), cand[3:4]])
            if has_intercept
            else np.log(cand)
        )

        res = minimize(
            objective,
            start_vec,
            method=init_method,
            options={"maxiter": init_iter, **options_override},
            **minimize_kwargs,
        )  # type: ignore[call-overload]

        resid_norm = res.fun
        if not np.isnan(resid_norm) and (best is None or np.isnan(best[0]) or resid_norm < best[0]):
            best = (resid_norm, res)

    if best is None:
        raise RuntimeError(
            f"No candidate start converged for kappa={kappa}, gamma={signal_strength}"
        )
    res_final = best[1]

    # solution needs to be transformed back from log-space to real-space
    soln = (
        np.concatenate([np.exp(res_final.x[:3]), res_final.x[3:4]])
        if has_intercept
        else np.exp(res_final.x)
    )

    if has_intercept: 
        mu, b, sigma, iota = soln 
    else: 
        mu, b, sigma = soln 
        iota = None
    state_params = StateParameters(mu=mu, b=b, sigma=sigma, iota = iota)

    return SolverResult(
        solution=state_params,
        func_value=g(res_final.x),
        message=res_final.message,
        success=res_final.success,
    )


def _root_solver(
    kappa: float,
    signal_strength: float,
    alpha: float,
    start: NDArray[np.float64],
    main_method: str = "hybr",
    gh: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    transform: bool = True,
    intercept: float | None = None,
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
    signal_strength : float
        Square root of signal strength or of corrupted signal
        strength, depending on whether `corrupted = TRUE` or not. If corrupted is
        False, then `signal_strength` is the limit `gamma` squared of the var(X * beta). If
        corrupted is True, then `signal_strength` is the limit `nu` squared of
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
        If False, then `signal_strength` is the square root of the signal strength.
        If True, then `signal_strength` is the square root of the corrupted signal strength
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
    has_intercept = intercept is not None

    gh = gh if gh is not None else _get_hermite_roots_weights(200)

    if corrupted:
        g = _se_funcs(
            kappa, signal_strength, alpha, gh, prox_tol, corrupted, transform, iota=intercept
        )
    else:
        g = _se_funcs(
            kappa, signal_strength, alpha, gh, prox_tol, corrupted, transform, intercept=intercept
        )

    if transform:
        start_t = (
            np.concatenate([np.log(start[:3]), start[3:4]])
            if has_intercept
            else np.log(start)
        )
    else:
        start_t = start

    res = root(g, start_t, method=main_method, **root_kwargs)  #  type: ignore[call-overload]

    if transform:
        soln = (
            np.concatenate([np.exp(res.x[:3]), res.x[3:4]])
            if has_intercept
            else np.exp(res.x)
        )
    else:
        soln = res.x

    if has_intercept: 
        mu, b, sigma, iota = soln 
    else: 
        mu, b, sigma = soln 
        iota = None
    state_params = StateParameters(mu=mu, b=b, sigma=sigma, iota = iota)
     
    return SolverResult(
        solution= state_params, func_value=g(res.x), message=res.message, success=res.success
    )


def solve_state_equation(
    kappa: float,
    signal_strength: float,
    alpha: float,
    start: NDArray[np.float64] | None = None,
    gh: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    root_kwargs: dict[str, Any] | None = None,
    minimize_kwargs: dict[str, Any] | None = None,
    transform: bool = True,
    corrupted: bool = False,
    intercept: float | None = None,
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
    signal_strength : float
        Square root of signal strength or of corrupted signal
        strength, depending on whether `corrupted = TRUE` or not. If corrupted is
        False, then `signal_strength` is the limit `gamma` squared of the var(X * beta). If
        corrupted is True, then `signal_strength` is the limit `nu` squared of
        \\text{var}(X * \\hat \\beta), where \\hat{\\beta} is the maximum
        Diaconis-Ylvisaker prior penalized likelihood (MDYPL) estimator as
        computed by [mdyplFit()] with shrinkage parameter alpha.
    alpha : float
        Shrinkage parameter of the MDYPL estimator. `alpha` should be in (0,1).
    start : NDArray[np.float64] | None
        A 1D array of length 3 containing the initial starting values for `mu`, `b`,
        and `sigma`. By default, None.
    gh : tuple[NDArray[np.float64], NDArray[np.float64]] | None
        A tuple of 1D arrays containing Gauss-Hermite quadrature nodes and weights
        used to approximate the system's expected values. By default, None.
    root_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed directly to the main root solver
        (`scipy.optimize.root`).
    minimize_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed directly to the initial minimizer
        (`scipy.optimize.minimize`).
    corrupted: bool, optional
        If False, then `signal_strength` is the square root of the signal strength.
        If True, then `signal_strength` is the square root of the corrupted signal strength
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
    has_intercept = intercept is not None

    # Intialise start as default brglm2 guess if None
    if start is None:
        start = (
            np.asarray([0.5, 1, 1], dtype=float)
            if not has_intercept
            else np.asarray([0.5, 1, 1, 0], dtype=float)
        )
    else: 
        _validate_start(start, has_intercept)

    root_kwargs = root_kwargs or {}
    minimize_kwargs = minimize_kwargs or {}

    if init_iter > 0:
        init_result = _init_solver(
            kappa,
            signal_strength,
            alpha,
            start,
            init_method,
            init_iter,
            gh,
            prox_tol,
            corrupted,
            intercept,
            **minimize_kwargs,
        )
        soln = init_result.solution
        start = soln.to_array()
        opt_chain = f"initial_method: {init_method} -> "
    else:
        opt_chain = ""

    result = _root_solver(
        kappa,
        signal_strength,
        alpha,
        start,
        main_method,
        gh,
        prox_tol,
        corrupted,
        transform,
        intercept,
        **root_kwargs,
    )
    opt_chain += f"main_method: {main_method}"

    return result, opt_chain
