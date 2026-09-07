import warnings
from collections.abc import Callable
from enum import IntEnum
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize, root

from to_be_titled import state_equations, validation
from to_be_titled.inference import derive_gamma_from_nu
from to_be_titled.interpolators.build_interpolator import _build_rgi_pchip_interpolator
from to_be_titled.solvers.solver_types import SolverResult, StateParameters
from to_be_titled.types import FloatArray


class SolverConvergenceError(RuntimeError):
    """Raised when every strategy in the solve cascade fails
    to converge to a valid, in-domain root."""


class AdaptiveAlphaMismatchWarning(RuntimeWarning):
    """Raised when use_warm_start_interpolator=True but alpha does not
    match the adaptive shrinkage value the interpolator's reference grid
    was built on."""


class ConvergenceCode(IntEnum):
    DID_NOT_CONVERGE = 0
    CONVERGED_FIRST_TRY = 1
    CONVERGED_AFTER_INIT = 2


def _transform_parameters(
    pars: FloatArray, has_intercept: bool, reverse: bool = False
) -> FloatArray:
    """
    If `reverse = False`, parameters are assumed to be in real-space and
    `mu`, `b`, `sigmas` and are subsequently transformed into log-space.
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


def _default_start(intercept: float | None = None) -> FloatArray:
    """
    Single arbitrary starting guess for `mu`, `b`, `sigma` and
    optional `intercept` when the user supplies no start.
    """
    has_intercept = intercept is not None
    return (
        np.asarray([0.5, 2.0, 2.0, intercept], dtype=np.float64)
        if has_intercept
        else np.asarray([0.5, 2.0, 2.0], dtype=np.float64)
    )


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
    Returns a closure compatible with scipy.optimize.root and scipy.optimize.minimize.
    The closure evaluates the three (or four) state evolution equations
    using Gauss–Hermite quadrature and optionally performs a log-transformation
    of the optimisation variables to enforce strict positivity and improve
    convergence.

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
        \nu (square root of the limit of Var(X * hat{\beta}) estimated via the
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
            pars_t = _transform_parameters(pars, has_intercept, reverse=True)
        else:
            pars_t = pars
        mu, b, sigma = pars_t[:3]

        if corrupted:
            gamma = derive_gamma_from_nu(kappa, signal_strength, sigma, mu)
            if np.isnan(gamma):
                warnings.warn(
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
                # iota is being solved
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
        # Estimated sample intercept is given from MLE
        else:
            if intercept is not None:
                # attempting to solve for theta_0
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


def _init_solver(
    kappa: float,
    signal_strength: float,
    alpha: float,
    start: FloatArray,
    init_method: str = "Nelder-Mead",
    hermite_roots_weights: tuple[FloatArray, FloatArray] | None = None,
    prox_tol: float = 1e-10,
    corrupted: bool = False,
    intercept: float | None = None,
    init_iter: int | None = None,
    **minimize_kwargs: Any,
) -> SolverResult:
    r"""
    The initialisation solver aims to minimise the sum of squared residuals
    of the MDYPL state-evolution equations. Root-finding algorithms can be
    sensitive to starting values, hence the goal of this intial minimisation
    algorithm is to find a starting vector, closer to the true roots of the
    state-equations to be fed into a root-finding algorithm.

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
        Shrinkage parameter of the MDYPL estimator. `alpha` should be in (0,1).
    start : FloatArray
        Initial starting values for the state evolution parameters. If `intercept`
        is None, a 1D array of length 3 containing (`mu`, `b`, `sigma`), where `mu`
        lies in (0, 1) and `b`, `sigma` are strictly positive. If `intercept` is
        provided, a 1D array of length 4 containing (`mu`, `b`, `sigma`, `iota`).
        These are internally log-transformed during minimization to enforce strict
        positivity and prevent underflow
    init_method : str, optional
        The optimization method passed into `scipy.optimize.minimize`
        to minimize the sum of squared residuals, by default 'Nelder-Mead'.
    init_iter : int | None, optional
        Maximum number of iterations for the initial minimization algorithm.
        By default, None and scipy.optimize.minimize() default used.
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
    if init_iter is not None:
        options_override = {"maxiter": init_iter, **options_override}

    # Log-transform parameters except `intercept` if exists
    start_t = _transform_parameters(start, has_intercept, reverse=False)

    res = minimize(
        objective,
        start_t,
        method=cast(Any, init_method),
        options=cast(Any, options_override),
        **minimize_kwargs,
    )  # pyright: ignore[reportCallIssue]

    soln = _transform_parameters(res.x, has_intercept, reverse=True)

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
        func_value=g(res.x),
        message=res.message,
        success=res.success,
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
    Executes a root-finding algorithm to estimate the stationary point
    (`mu*`, `b*`, `sigma*`) and an optional `iota`/`theta` of the MDYPL
    state evolution equations.

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
        A 1D array of initial starting values for the state evolution parameters.
        If `intercept` is None, length must be 3 containing (`mu`, `b`, `sigma`). If
        `intercept` is provided, length must be 4 containing (`mu`, `b`, `sigma`,
        `iota`). For extreme `kappa` and `gamma` regimes, it is recommended to pass a
        "warm" starting guess obtained from an initial heuristic solver.
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
    *,
    use_warm_start_interpolator: bool = True,
    hermite_roots_weights: tuple[FloatArray, FloatArray] | None = None,
    root_kwargs: dict[str, Any] | None = None,
    minimize_kwargs: dict[str, Any] | None = None,
    transform: bool = True,
    corrupted: bool = False,
    intercept: float | None = None,
    init_method: str = "Nelder-Mead",
    main_method: str = "hybr",
    init_iter: int | None = None,
    prox_tol: float = 1e-10,
    convergence_tol: float = 1e-4,
    warn_interp_alpha_mismatch: bool = True,
) -> tuple[SolverResult, ConvergenceCode]:
    r"""
    Solves the MDYPL state equations.

    This wrapper function executes a root-finding algorithm to locate the stationary
    points of the 3-4 MDYPL state equations. As root-finding algorithms can be sensitive
    to the initial start, if it is unsucessful the function executes
    `scipy.optimize.minimize` with the previous initial start to attempt to minimize
    the sum of squared residuals of the state equations. The solution of this minimizer
    is used as the "warm" start for the root-finding algorithm for a second time.

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
        Shrinkage parameter of the MDYPL estimator. `alpha` should be in [0,1].
    start : FloatArray | None, optional
        Initial values for the state evolution parameters. If `intercept` is None,
        `start` must be a 1D array of length 3 containing (`mu`, `b`, `sigma`),
        where `mu` lies in (0, 1), and `b` and `sigma` are strictly positive. If
        `intercept` is provided, `start` must be a 1D array of length 4 containing
        (`mu`, `b`, `sigma`, `iota`). If `use_warm_start_interpolator` is False,
        this defaults to `[0.5, 2.0, 2.0]` when `intercept` is None, or
        `[0.5, 2.0, 2.0, intercept]` when an intercept is specified,
        by default None.
    use_warm_start_interpolator : bool, optional
        A PCHIP `RegularGridInterpolator` is fit on a grid of 100x100 approximate
        true values of the state equations in the uncorrupted case without an
        intercept. If `use_warm_start_interpolator` is True, the true parameter
        values are interpolated using `kappa` and `signal_strength` and is used as
        default start. By default, True.
    hermite_roots_weights : tuple[FloatArray, FloatArray] | None, optional
        A tuple of 1D arrays containing Gauss-Hermite quadrature nodes and weights
        used to approximate the system's expected values. Defaults to None, wherein
        200 nodes are used to approximate bivariate integrals within state equations.
    root_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed directly to the main root solver
        (`scipy.optimize.root`). By default None
    minimize_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed directly to the initial minimizer
        (`scipy.optimize.minimize`), by default None
    transform : bool, optional
        If True, the input parameters (`mu`, `b`, `sigma`) are internally
        log-transformed before being passed into scipy.optimize.root().
        This enforces strict positivity and can improve convergence.
        By default, True.
    corrupted : bool, optional
        If False, `signal_strength` is the true signal strength \gamma.
        If True, `signal_strength` is the corrupted signal strength, \nu.
        Default is False
    intercept : float | None, optional
        If None, the function minimizes residuals for the 3-equation system
        without an intercept (`mu`,`b`, `sigma`).
        If a float:
        - If `corrupted = False`, `intercept` represents the true population
        intercept \theta_0, which is appended as the intial guess for `iota`
        if `start` is None.
        - If `corrupted = True`, `intercept` represents the limit, `iota`, of the
        MDYPL sample-estimated intercept \hat{\theta}_0, which is appended as the
        initial guess for `theta0` if `start` is None.
    init_method : str, optional
        The optimization method to be passed to `scipy.optimize.minimize` for the
        initial warm-start phase, by default "Nelder-Mead"
    main_method : str, optional
        The root-finding method to be passed to `scipy.optimize.root` for the exact
        solution phase, by default "hybr".
    init_iter : int | None, optional
        Maximum number of iterations for the initial minimization algorithm if needed.
        By default, None and scipy.optimize.minimize() default used.
    prox_tol : float, optional
        Convergence tolerance for the Newton-Raphson estimation of the
        proximal operator, by default 1e-10.
    convergence_tol: float, optional
        Convergence tolerance used to assess final solution. Enforces
        that func(solution) is less than `convergence_tol`. By default,
        1e-4.
    warn_interp_alpha_mismatch: bool, optional
        If True, and `use_warm_start_interpolator=True`, warns via
        `AdaptiveAlphaMismatchWarning` when `alpha` does not closely match
        the adaptive shrinkage value `1/(1+kappa)` that the interpolator's
        reference grid was built on. The interpolated warm start is
        only technically valid for that adaptive alpha,
        and may be a weaker starting guess for other values of `alpha`.
        Set to False to silence this warning without changing solver
        behaviour. By default, True.

    Returns
    -------
    tuple[SolverResult, str]
        A two-element tuple containing:
        - SolverResult: dataclass with the optimal real-space parameters
          (`solution`), the evaluated residual vector (`func_value`)and
          the main solver's converges success/message.
        - ConvergenceCode: an IntEnum indicating which stage of the solve
          cascade succeeded (or that none did):
          `DID_NOT_CONVERGE` (0), `CONVERGED_FIRST_TRY` (1), or
          `CONVERGED_AFTER_INIT` (2).
    Raises
    ------
    SolverConvergenceError
        If every strategy in the solve cascade fails to converge to a valid,
        in-domain root.
    """
    has_intercept = intercept is not None

    validation.validate_state_equation_fixed_params(alpha, kappa, signal_strength)
    if start is not None:
        validation.validate_start_dims(start, has_intercept)
        validation._validate_domain(start)

    root_kwargs = root_kwargs or {}
    minimize_kwargs = minimize_kwargs or {}

    attempts: list[tuple[str, SolverResult]] = []

    # --- Stage 1: Pick Initial Start --
    if start is not None:
        stage1_start, stage1_name = start, "user_start"
    elif use_warm_start_interpolator:
        interp = _build_rgi_pchip_interpolator()

        expected_alpha = 1 / (1 + kappa)
        if (
            not np.isclose(alpha, expected_alpha, rtol=1e-2)
            and warn_interp_alpha_mismatch
        ):
            warnings.warn(
                "alpha does not match the adaptive shrinkage value 1/(1+kappa); "
                "the interpolated warm start may be less reliable. See "
                "solve_state_equation docs for details.",
                AdaptiveAlphaMismatchWarning,
                stacklevel=2,
            )

        if not corrupted:
            stage1_start = interp.evaluate(kappa, signal_strength)
        else:
            start_temp = interp.evaluate(kappa, signal_strength)
            # get estimate for gamma, given kappa, corrupted ss
            # and interpolated values for `mu` and `sigma`
            gamma_est = derive_gamma_from_nu(
                kappa, signal_strength, start_temp[2], start_temp[0]
            )
            stage1_start = interp.evaluate(kappa, gamma_est)

        if intercept is not None:
            # use iota/theta as guess for theta/iota
            stage1_start = np.append(stage1_start, intercept)

        stage1_name = "interpolated_start"
    else:
        stage1_start, stage1_name = _default_start(intercept), "default_start"

    def try_root(candidate_start: FloatArray) -> SolverResult:
        return _root_solver(
            kappa,
            signal_strength,
            alpha,
            candidate_start,
            main_method,
            hermite_roots_weights,
            prox_tol,
            corrupted,
            transform,
            intercept,
            **root_kwargs,
        )

    result = try_root(stage1_start)
    attempts.append((stage1_name, result))

    if validation.is_valid(result, tol=convergence_tol):
        return result, ConvergenceCode.CONVERGED_FIRST_TRY

    ## -- Stage 2: Fallback method supplying stage1_start inot _init_solver --
    init_result = _init_solver(
        kappa,
        signal_strength,
        alpha,
        stage1_start,
        init_method,
        hermite_roots_weights,
        prox_tol,
        corrupted,
        intercept,
        init_iter,
        **minimize_kwargs,
    )
    warm_start = init_result.solution.to_array()

    result = try_root(warm_start)
    attempts.append((f"{init_method}_warm_start", result))

    if validation.is_valid(result, tol=convergence_tol):
        return (result, ConvergenceCode.CONVERGED_AFTER_INIT)

    warnings.warn(
        f"All strategies failed to converge at kappa={kappa}, "
        f"signal_strength={signal_strength}. Returning last (unvalidated) solution. "
        f"Attempts: {[(name, r.success) for name, r in attempts]}. "
        "Try alternative start or allow for interpolation warm start.",
        RuntimeWarning,
        stacklevel=2,
    )

    return (result, ConvergenceCode.DID_NOT_CONVERGE)
