from collections.abc import Callable
from typing import Any, cast
from warnings import warn

import numpy as np
from scipy.optimize import OptimizeResult, minimize, root

from to_be_titled import state_equations, validation
from to_be_titled.solvers.solver_types import SolverResult, StateParameters
from to_be_titled.types import FloatArray


def _se_funcs(
    kappa: float,
    signal_strength: float,
    alpha: float,
    hermite_roots_weights: tuple[FloatArray, FloatArray] | None,
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    transform: bool = True,
    intercept: float | None = None,
) -> Callable[[FloatArray], FloatArray]:
    r"""
    Construct the system of MDYPL state evolution equations.
    Returns a closure compatible with scipy.optimize.root and scipy.optimize.minimize.
    The closure evaluates the three (or four) state evolution equations
    using Gauss–Hermite quadrature and optionally performs a log-transformation
    of the optimisation variables to enforce positivity.

    Parameters
    ----------
    kappa : float
        Asymptotic ratio of the columns/rows of the design matrix (p/n).
        kappa should be in (0,1).
    signal_strength : float
        Square root of the signal strength (ss).
        - If `corrupted = False`, represents the true signal strength
        \gamma (square root of the limit of Var(X * \beta_0))
        - If `corrupted = True`, represents the corrupted signal strength
        \nu (square root of the limit of Var(X * \hat{\beta}) estimated via the
        signal strength leave one out estimator).
    alpha : float
        Shrinkage hyperparamter of the MDYPL estimator. `alpha` should be in `(0,1]`.
    hermite_roots_weights : tuple[FloatArray, FloatArray]
        Tuple of 1D arrays containing Gauss Hermite quadrature nodes and weights
        used to approximate the expected values in the state equations.
    prox_tol : float, optional
        Convergence tolerance for the Newton-Raphson estimation of
        proximal operator. Default is 1e-10.
    corrupted : bool, optional
        If False, `signal_strength` is the true signal strength \gamma.
        If True, `signal_strength` is the corrupted signal strength, \nu.
        Default is False.
    transform : bool, optional
        If True, the closure expects the input parameters (`mu`, `b`, `sigma`)
        to be log-transformed. The closure will automatically clip and exponentiate
        the inputs to enforce strict positivity and prevent numerical
        underflow/overflow during solver exploration. By default True.
    intercept: float | None, optional
        If `intercept` is None, then the 3-equation system without an intercpet is
        used. If `intercept` is a float, then when
        - `corrupted = False`, `intercept` specifies the known (fixed) population
        intercept (\theta_0) used when solving the 4-equation state evolution system,
        for `iota` which is the asymptotic limit of the sample estimated intercept.
        - `corrupted = True`, `intercept` specifies the (fixed) sample intercept
        estimate (\hat{\theta}_0) used to solve the 4-equation state evolution system,
        for the true intercept \theta_0.

    Returns
    -------
    Callable[[FloatArray], FloatArray]
        An objective function `g(pars)` that takes a 1D array of the three state
        evolution parameters (optionally log-transformed) and returns a 1D array
        of the evaluated residuals for the three state equations.
    """

    def g(pars: FloatArray) -> FloatArray:
        """
        The residual closure function passed into scipy solvers to be optimised.
        """
        pars = np.asarray(pars, dtype=np.float64)
        has_intercept = intercept is not None

        if transform:
            # Transform (`mu`,`b`, `sigma`) parameters back into real-space
            pars_t = _transform_parameters(pars, has_intercept, reverse=True)
        else:
            pars_t = pars
        mu, b, sigma = pars_t[:3]

        def _derive_gamma_from_nu() -> float:
            with np.errstate(invalid="ignore"):
                gamma = np.sqrt(signal_strength**2 - kappa * sigma**2) / mu
            return float(gamma)

        if corrupted:
            gamma = _derive_gamma_from_nu()
            # Check estimated gamma did not evaluate to NaN
            if np.isnan(gamma):
                warn(
                    "Estimated gamma evaluated to NaN (likely due to division by "
                    "zero). Returning NaNs for state equation residuals.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return np.full(4 if has_intercept else 3, np.nan)
        else:
            gamma = signal_strength

        # Case 1: Oracle case, true population intercept known
        if not corrupted:
            if intercept is not None:
                # iota is one of the variables in which we are trying to solve for
                iota_var = pars_t[3]
                theta_fixed = intercept

                return state_equations.se_with_intercept(
                    mu=mu,
                    b=b,
                    sigma=sigma,
                    iota=iota_var,
                    kappa=kappa,
                    gamma=gamma,
                    alpha=alpha,
                    theta=theta_fixed,
                    hermite_roots_weights=hermite_roots_weights,
                    prox_tol=prox_tol,
                )
            else:
                return state_equations.se_no_intercept(
                    mu=mu,
                    b=b,
                    sigma=sigma,
                    kappa=kappa,
                    gamma=gamma,
                    alpha=alpha,
                    hermite_roots_weights=hermite_roots_weights,
                    prox_tol=prox_tol,
                )

        # Case 2: Empirical (corrupted) case; true population intercept not known
        # Estimated sample intercept is given
        else:
            if intercept is not None:
                # true theta is now what we are trying to solve for
                theta_var = pars_t[3]
                # iota, is fixed and specified by estimated sample intercept
                iota_fixed = intercept
                return state_equations.se_with_intercept(
                    mu=mu,
                    b=b,
                    sigma=sigma,
                    iota=iota_fixed,
                    kappa=kappa,
                    gamma=gamma,
                    alpha=alpha,
                    theta=theta_var,
                    hermite_roots_weights=hermite_roots_weights,
                    prox_tol=prox_tol,
                )
            else:
                return state_equations.se_no_intercept(
                    mu=mu,
                    b=b,
                    sigma=sigma,
                    kappa=kappa,
                    gamma=gamma,
                    alpha=alpha,
                    hermite_roots_weights=hermite_roots_weights,
                    prox_tol=prox_tol,
                )

    return g


def _transform_parameters(
    pars: FloatArray, has_intercept: bool, reverse: bool = False
) -> FloatArray:
    """
    If `reverse = False`, parameters are assumed to be in real-space and
    `mu`, `b`, `sigmas` are subsequently transformed into log-space.
    If `reverse = True` then `mu`, `b`, and `sigma` (not iota) are
    assumed to be in log-space, and subsequently clipped and exponetiated
    back into real-space.
    """
    if not reverse:
        pars_log = (
            np.concatenate([np.log(pars[:3]), pars[3:4]])
            if has_intercept
            else np.log(pars)
        )
        return pars_log
    else:
        pars_clipped = pars.copy()
        pars_clipped[:3] = np.clip(pars[:3], -20, 20)

        pars_exp = (
            np.concatenate([np.exp(pars_clipped[:3]), pars_clipped[3:4]])
            if has_intercept
            else np.exp(pars_clipped)
        )
        return pars_exp


def _default_start(has_intercept: bool) -> FloatArray:
    """
    Single starting guess for `mu`, `b`, `sigma` and
    optional `intercept` when the user supplies no start.
    """
    return (
        np.asarray([0.5, 2.0, 2.0, 0.0], dtype=np.float64)
        if has_intercept
        else np.asarray([0.5, 2.0, 2.0], dtype=np.float64)
    )


def _generate_warm_start_candidates(
    start: FloatArray | None, signal_strength: float, has_intercept: bool
) -> list[FloatArray]:
    """
    Internal function to create a list of possible warm starts for initial
    solver.
    """
    user_start = (
        np.asarray(start, dtype=np.float64)
        if start is not None
        else _default_start(has_intercept)
    )

    candidates: list[FloatArray] = [user_start]

    # TODO: Find best alternative candidates
    if has_intercept:
        candidates.extend(
            [
                np.array(
                    [0.5, signal_strength, signal_strength, 0.0]
                ),  # state equation-scaled default
                np.array([0.1, 1.0, 1.0, 0.0]),  # low-mu fallback
                np.array(
                    [0.9, signal_strength * 2, signal_strength, 0.0]
                ),  # large mu fallback
            ]
        )
    else:
        candidates.extend(
            [
                np.array(
                    [0.5, signal_strength, signal_strength]
                ),  # state equation-scaled default
                np.array([0.1, 1.0, 1.0]),  # low-mu fallback
                np.array(
                    [0.9, signal_strength * 2, signal_strength]
                ),  # large mu fallback
            ]
        )

    return candidates


def _init_solver(
    kappa: float,
    signal_strength: float,
    alpha: float,
    start: FloatArray | None = None,
    init_method: str = "Nelder-Mead",
    init_iter: int = 50,
    hermite_roots_weights: tuple[FloatArray, FloatArray] | None = None,
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    intercept: float | None = None,
    **minimize_kwargs: Any,
) -> SolverResult:
    r"""
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
        Square root of the signal strength (ss).
        - If `corrupted = False`, represents the true signal strength
        \gamma (square root of the limit of Var(X * \beta_0))
        - If `corrupted = True`, represents the corrupted signal strength
        \nu (square root of the limit of Var(X * \hat{\beta}) estimated via the
        signal strength leave one out estimator).
    alpha : float
        Shrinkage parameter of the MDYPL estimator. `alpha` should be in `(0,1]`.
    start : FloatArray, optional
        A 1D array of starting values with (`mu`, `b`, `sigma`) with an optional,
        `iota`/`theta` value if an intercept is included. The first three parameters
        are internally log-transformed during minimization to enforce strict
        positivity and prevent underflow. If None, default candidate starting vectors
        are used.
    init_method : str, optional
        The optimization method passed into `scipy.optimize.minimize`
        to minimize the sum of squared residuals, by default 'Nelder-Mead'.
    init_iter : int, optional
        Maximum number of iterations for the initial minimization algorithm,
        by default 50.
    hermite_roots_weights : tuple[FloatArray, FloatArray] | None, optional
        A tuple of 1D arrays containing Gauss-Hermite quadrature nodes and weights
        used to approximate the system's expected values. By default None, in
        which case it is set to `scipy.special.roots_hermite(200)`.
    prox_tol : float, optional
        Convergence tolerance for the Newton-Raphson estimation of the
        proximal operator, by default 1e-10.
    corrupted: bool, optional
        If False, `signal_strength` is the true signal strength \gamma.
        If True, `signal_strength` is the corrupted signal strength, \nu.
        Default is False.
    intercept : float | None, optional
        If None, the function minimizes residuals for the 3-equation system
        without an intercept (`mu`,`b`, `sigma`).
        If a float:
        - If `corrupted = False`, `intercept` represents the true population
        intercept \theta_0.
        - If `corrupted = True`, `intercept` represents the limit, `iota`, of the
        MDYPL sample-estimated intercept \hat{\theta}_0.
    **minimize_kwargs : dict[str, Any], optional
        Additional keyword arguments passed directly to `scipy.optimize.minimize`.

    Returns
    -------
    SolverResult
        A dataclass containing the optimal real-space parameters (`solution`),
        the evaluated residual vector at the solution (`func_value`), and the
        solver's convergence status (`message`, `success`).
    """
    has_intercept = intercept is not None

    # Appends additional warm start candidates to (optional) user start
    # Allows for _init_solver to find the best possible warm start.
    candidates = _generate_warm_start_candidates(start, signal_strength, has_intercept)

    best: tuple[float, OptimizeResult] | None = None

    g = _se_funcs(
        kappa=kappa,
        signal_strength=signal_strength,
        alpha=alpha,
        hermite_roots_weights=hermite_roots_weights,
        prox_tol=prox_tol,
        corrupted=corrupted,
        transform=True,
        intercept=intercept,
    )

    def objective(pars_log: FloatArray) -> float:
        """
        Defines the objective to be passed into scipy.optimize.minimize, which is
        to minimize the sum of squared residuals.
        """
        r = g(pars_log)
        return float(np.dot(r, r))

    options_override = minimize_kwargs.pop("options", {})

    for cand in candidates:
        cand = np.asarray(cand, dtype=np.float64)

        # Log-transform parameters except `intercept` if exists
        cand_t = _transform_parameters(cand, has_intercept, reverse=False)

        res = minimize(
            objective,
            cand_t,
            method=cast(Any, init_method),
            options=cast(Any, {"maxiter": init_iter, **options_override}),
            **minimize_kwargs,
        )  # pyright: ignore[reportCallIssue]

        resid_norm = res.fun
        soln_raw = _transform_parameters(res.x, has_intercept, reverse=True)

        # Ensure roots found lie within valid domain
        # Otherwise no sense to pass in as warm start to _root_solver
        if not validation._is_valid_domain(soln_raw):
            continue

        if (
            not np.isnan(resid_norm)
            and res.success
            and (best is None or resid_norm < best[0])
        ):
            best = (resid_norm, res)

    if best is None:
        raise RuntimeError(
            f"No candidate start converged for kappa={kappa}, gamma={signal_strength}"
        )

    # transform best solution back into real space from log-space
    res_best = best[1]
    soln_best = _transform_parameters(res_best.x, has_intercept, reverse=True)

    if has_intercept:
        mu, b, sigma, intercept_est = soln_best
        state_params = StateParameters(
            mu=mu,
            b=b,
            sigma=sigma,
            intercept_estimate=intercept_est,
            corrupted=corrupted,
        )
    else:
        mu, b, sigma = soln_best
        state_params = StateParameters(mu=mu, b=b, sigma=sigma, corrupted=corrupted)

    return SolverResult(
        solution=state_params,
        func_value=g(res_best.x),
        message=res_best.message,
        success=res_best.success,
    )


def _root_solver(
    kappa: float,
    signal_strength: float,
    alpha: float,
    start: FloatArray,
    main_method: str = "hybr",
    hermite_roots_weights: tuple[FloatArray, FloatArray] | None = None,
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    transform: bool = True,
    intercept: float | None = None,
    **root_kwargs: Any,
) -> SolverResult:
    r"""
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
        Square root of the signal strength (ss).
        - If `corrupted = False`, represents the true signal strength
        \gamma (square root of the limit of Var(X * \beta_0))
        - If `corrupted = True`, represents the corrupted signal strength
        \nu (square root of the limit of Var(X * \hat{\beta}) estimated via the
        signal strength leave one out estimator).
    alpha : float
        Shrinkage hyperparameter of the MDYPL estimator. `alpha` should be in `(0,1]`.
    start : FloatArray
        A 1D array of starting values with (`mu`, `b`, `sigma`) with an optional,
        `iota`/`theta` value if an intercept is included. Especially, for exteme
        `kappa` and `gamma` reigmes, it is reccomended to use an initial sum of
        squares minimizer to obtain a warm start for the root solver.
    main_method : str, optional
        The method to be passed into `scipy.optimize.root` to find the roots of
        the state equations, by default "hybr" .
    hermite_roots_weights : tuple[FloatArray, FloatArray] | None, optional
        A tuple of 1D arrays containing Gauss-Hermite quadrature nodes and weights
        used to approximate the system's expected values. By default, None.
    prox_tol : float, optional
        Convergence tolerance for the Newton-Raphson estimation of the
        proximal operator, by default 1e-10.
    corrupted: bool, optional
        If False, `signal_strength` is the true signal strength \gamma.
        If True, `signal_strength` is the corrupted signal strength, \nu.
        Default is False.
    intercept : float | None, optional
        If None, the function minimizes residuals for the 3-equation system
        without an intercept (`mu`,`b`, `sigma`).
        If a float:
        - If `corrupted = False`, `intercept` represents the true population
        intercept \theta_0.
        - If `corrupted = True`, `intercept` represents the limit, `iota`, of the
        MDYPL sample-estimated intercept \hat{\theta}_0.
    **root_kwargs : dict[str, Any], optional
        Additional keyword arguments passed directly to `scipy.optimize.root`.

    Returns
    -------
    SolverResult
        A dataclass containing the optimal real-space parameters (`solution`),
        the evaluated residual vector at the solution (`func_value`) and the
        solver's termination status and message (`message`, `success`).
    """
    has_intercept = intercept is not None

    g = _se_funcs(
        kappa,
        signal_strength,
        alpha,
        hermite_roots_weights,
        prox_tol,
        corrupted,
        transform,
        intercept=intercept,
    )

    if transform:
        start_t = _transform_parameters(start, has_intercept, reverse=False)
    else:
        start_t = start.copy()

    res = root(g, start_t, method=cast(Any, main_method), **root_kwargs)
    raw_x = np.asarray(res.x, dtype=np.float64)

    if transform:
        soln = _transform_parameters(raw_x, has_intercept, reverse=True)
    else:
        soln = raw_x

    # Domain validity check
    validation._validate_domain(soln)
    # Convergence check
    if not res.success:
        raise RuntimeError(
            f"Root solver did not converge using {main_method}"
            " with (kappa, ss, alpha)"
            f"= ({kappa},{signal_strength},{alpha})"
        )

    if has_intercept:
        mu, b, sigma, intercept_est = soln
        state_params = StateParameters(
            mu=mu,
            b=b,
            sigma=sigma,
            intercept_estimate=intercept_est,
            corrupted=corrupted,
        )
    else:
        mu, b, sigma = soln
        state_params = StateParameters(mu=mu, b=b, sigma=sigma, corrupted=corrupted)

    return SolverResult(
        solution=state_params,
        func_value=g(raw_x),
        message=res.message,
        success=res.success,
    )


def solve_state_equation(
    kappa: float,
    signal_strength: float,
    alpha: float,
    start: FloatArray | None = None,
    hermite_roots_weights: tuple[FloatArray, FloatArray] | None = None,
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
    r"""
    Solves the MDYPL state evolution equations.

    This wrapper function executes the optional initial sum of squares minimizer to
    obtain a warm start followed by a precise numerical root-finder.
    It provides a robust mechanism for finding the stationary points
    `mu*`, `b*`, and `sigma*` (and `iota`/`theta` if an intercept is included)
    of the system's state equations.

    Parameters
    ----------
    kappa : float
        Asymptotic ratio of the columns/rows of the design matrix (p/n).
        `kappa` should be in (0,1).
    signal_strength : float
        Square root of the signal strength (ss).
        - If `corrupted = False`, represents the true signal strength
        \gamma (square root of the limit of Var(X * \beta_0))
        - If `corrupted = True`, represents the corrupted signal strength
        \nu (square root of the limit of Var(X * \hat{\beta}) estimated via the
        signal strength leave one out estimator).
    alpha : float
        Shrinkage hyperparameter of the MDYPL estimator. `alpha` should be in `(0,1]`.
    start : FloatArray | None
        A 1D array of starting values with (`mu`, `b`, `sigma`) with an optional,
        `iota`/`theta` value if an intercept is included. By default, None.
    hermite_roots_weights : tuple[FloatArray, FloatArray] | None, optional
        A tuple of 1D arrays containing Gauss-Hermite quadrature nodes and weights
        used to approximate the system's expected values. By default, None.
    root_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed directly to the main root solver
        (`scipy.optimize.root`).
    minimize_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed directly to the initial minimizer
        (`scipy.optimize.minimize`).
    corrupted: bool, optional
        If False, `signal_strength` is the true signal strength \gamma.
        If True, `signal_strength` is the corrupted signal strength, \nu.
        Default is False.
    intercept: float | None, optional
        If None, the function minimizes residuals for the 3-equation system
        without an intercept (`mu`,`b`, `sigma`).
        If a float:
        - If `corrupted = False`, `intercept` represents the true population
        intercept \theta_0.
        - If `corrupted = True`, `intercept` represents the limit, `iota`, of the
        MDYPL sample-estimated intercept \hat{\theta}_0.
    transform : bool, optional
        If True, the input parameters (`mu`, `b`, `sigma`) are internally
        log-transformed before being passed into scipy.optimize.root().
        This enforces strict positivity and can improve convergence.
        By default, True.
    init_iter : int, optional
        The number of iterations to run the initial minimization algorithm. If
        `init_iter` is greater than zero, the result from `_init_solver` is passed
        into `_root_solver` as a warm start. By default, 50.
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
          (`solution`), the evaluated residual vector (`func_value`)and
          the main solver's converges success/message.
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

    validation._validate_state_equation_fixed_params(alpha, kappa, signal_strength)

    if start is not None:
        validation._validate_start_dims(start, has_intercept)
        validation._validate_domain(start)

    root_kwargs = root_kwargs or {}
    minimize_kwargs = minimize_kwargs or {}

    if init_iter > 0:
        try:
            init_result = _init_solver(
                kappa,
                signal_strength,
                alpha,
                start,
                init_method,
                init_iter,
                hermite_roots_weights,
                prox_tol,
                corrupted,
                intercept,
                **minimize_kwargs,
            )
            soln = init_result.solution
            start = soln.to_array()
            opt_chain = f"initial_method: {init_method} -> "

        # If no candidate start converged for _init_solver, use user supplied
        # or default start straight into root solver
        except RuntimeError:
            if start is None:
                start = _default_start(has_intercept)
            opt_chain = f"initial_method: {init_method} (failed, fallback to start) -> "

    else:
        if start is None:
            start = _default_start(has_intercept)
        opt_chain = ""

    result = _root_solver(
        kappa,
        signal_strength,
        alpha,
        start,
        main_method,
        hermite_roots_weights,
        prox_tol,
        corrupted,
        transform,
        intercept,
        **root_kwargs,
    )

    opt_chain += f"main_method: {main_method}"

    return result, opt_chain
