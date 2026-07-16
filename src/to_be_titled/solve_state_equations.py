from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, root
from scipy.special import roots_hermite

from to_be_titled import state_equations


def se_funcs(
    kappa: float,
    gamma: float,
    alpha: float,
    gh: tuple[NDArray[np.float64], NDArray[np.float64]],
    prox_tol: float = 1e-10,
    transform: bool = False,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """_summary_

    Parameters
    ----------
    kappa : float
        asymptotic ratio of the columns/rows of the design matrix.
        kappa should be in (0,1).
    gamma : float
        square root of signal strength.
    alpha : float
        shrinkage parameter of the MDYPL estimator. alpha should be in (0,1).
    gh : tuple[NDArray[np.float64], NDArray[np.float64]]
        a tuple of gauss hermite quadrature nodes and weights.
    prox_tol : float, optional
        tolerance for the computation of the proximal operator,
        by default 1e-10
    transform : bool, optional
        true if parameters have been log transformed, by default False.

    Returns
    -------
    Callable[[NDArray[np.float64]], NDArray[np.float64]]
        returns function which evaluates state equations for given
        kappa, gamma, alpha and gauss-hermite nodes and weights.
    """

    def g(pars: NDArray[np.float64]) -> NDArray[np.float64]:
        if transform:
            # Prevents underflow/overflow when exponentiating
            pars_clipped = np.clip(pars, -250, 250)
            pars = np.exp(pars_clipped)

        return state_equations._se0(
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
    solution: NDArray[np.float64]
    func_value: NDArray[np.float64]
    iterations: int
    message: str
    termination_code: int


def init_solver(
    kappa: float,
    gamma: float,
    alpha: float,
    start: NDArray[np.float64],
    init_method: str = "Nelder-Mead",
    gh: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    prox_tol: float = 1e-10,
    **minimize_kwargs: Any,
) -> SolverResult:
    """

    Initial solver to find warm start for main solver. The initial solver uses the
    specified method to minimise the sum of squares of the residuals of the state
    equations.

    Parameters
    ----------
    kappa : float
        asymptotic ratio of the columns/rows of the design matrix.
        kappa should be in (0,1).
    gamma : float
        square root of signal strength.
    alpha : float
        shrinkage parameter of the MDYPL estimator. alpha should be in (0,1).
    start : NDArray[np.float64]
        vector with starting values for `mu`, `b` and `sigma`.
    init_method : str, optional
        the method to be passed into scipy.minimize to minimize the sum of the
        squared residuals, by default 'Nelder-Mead'
    gh : tuple[NDArray[np.float64], NDArray[np.float64]] | None, optional
        a tuple of gauss hermite quadrature nodes and weights as returned by
        scipy.special.roots_hermite(), by default None in which case `gh`
        is set to scipy.special.roots_hermite(200).
    prox_tol : float, optional
        tolerance for the computation of the proximal operator, by default 1e-10.

    Returns
    -------
    SolverResult
        Dataclass containing the solution, the residual vector at the solution,
        the number of iterations, and the solver's convergence message/status.
    """
    if gh is None:
        gh = roots_hermite(200)

    g = se_funcs(kappa, gamma, alpha, gh=gh, prox_tol=prox_tol, transform=True)
    start_log = np.asarray(np.log(start), dtype=np.float64)

    def objective(pars_log: NDArray[np.float64]) -> float:
        r = g(pars_log)
        return float(np.dot(r, r))

    # TODO: Lewis see comment needed for this to pass mypy test
    res = minimize(objective, start_log, method=init_method, **minimize_kwargs)  # type: ignore[call-overload]

    soln = np.exp(res.x)

    return SolverResult(
        solution=soln,
        func_value=g(res.x),
        iterations=res.nit,
        message=res.message,
        termination_code=res.status,
    )


def root_solver(
    kappa: float,
    gamma: float,
    alpha: float,
    start: NDArray[np.float64],
    main_method: str = "hybr",
    gh: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    prox_tol: float = 1e-10,
    transform: bool = False,
    **root_kwargs: Any,
) -> SolverResult:
    """
    Main solver which uses scipy.optimize.root to find the root of the
    state equation.

    Parameters
    ----------
    kappa : float
        asymptotic ratio of the columns/rows of the design matrix.
        kappa should be in (0,1).
    gamma : float
        square root of signal strength.
    alpha : float
        shrinkage parameter of the MDYPL estimator. alpha should be in (0,1).
    start : NDArray[np.float64]
        vector with starting values for `mu`, `b` and `sigma`.
    main_method : str, optional
        the method to be passed into scipy.root to find the roots of the state
        equations, by default "hybr"
    gh : tuple[NDArray[np.float64], NDArray[np.float64]] | None, optional
         a tuple of gauss hermite quadrature nodes and weights as returned by
        scipy.special.roots_hermite(), by default None in which case `gh`
        is set to scipy.special.roots_hermite(200)., by default None
    prox_tol : float, optional
        tolerance for the computation of the proximal operator, by default 1e-10

    Returns
    -------
    SolverResult
        Dataclass containing the solution, the residual vector at the solution,
        the number of iterations, and the solver's convergence message/status.
    """

    if gh is None:
        gh = roots_hermite(200)

    g = se_funcs(kappa, gamma, alpha, gh, prox_tol, transform)

    start = np.log(start) if transform else start

    res = root(g, start, method=main_method, **root_kwargs)  #  type: ignore[call-overload]

    soln = np.exp(res.x) if transform else res.x  # Return in original space

    return SolverResult(
        solution=soln,
        func_value=g(res.x),
        iterations=res.nit,
        message=res.message,
        termination_code=res.status,
    )


def solve_se(
    kappa: float,
    gamma: float,
    alpha: float,
    start: NDArray[np.float64],
    root_kwargs: Any,
    minimize_kwargs: Any,
    transform: bool = True,
    init_iter: int | Literal["only"] = 50,
    init_method: str = "Nelder-Mead",
    main_method: str = "hybr",
    prox_tol: float = 1e-10,
    gh: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> SolverResult:
    """_summary_

    Parameters
    ----------
    kappa : float
        _description_
    gamma : float
        _description_
    alpha : float
        _description_
    start : NDArray[np.float64]
        _description_
    root_kwarg : Any
        _description_
    minimize_kwarg : Any
        _description_
    transform : bool, optional
        _description_, by default True
    init_iter : int, optional
        _description_, by default 50
    init_method : str, optional
        _description_, by default "Nelder-Mead"
    main_method : str, optional
        _description_, by default 'hybr'
    prox_tol : float, optional
        _description_, by default 1e-10
    gh : tuple[NDArray[np.float64], NDArray[np.float64]] | None, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        _description_
    ValueError
        _description_
    """

    npar = 3

    try:
        start_len = len(start)
    except TypeError:
        raise TypeError("`start` must be a sequence with a length")
    if start_len != npar:
        raise ValueError(f"start must have length {npar}; got {start_len}")

    root_kwargs = root_kwargs or {}
    minimize_kwargs = minimize_kwargs or {}

    if init_iter == "only":
        result = init_solver(
            kappa, gamma, alpha, start, init_method, gh, prox_tol, **minimize_kwargs
        )
    else:
        if init_iter > 0:
            init_result = init_solver(
                kappa, gamma, alpha, start, init_method, gh, prox_tol, **minimize_kwargs
            )
            warm_start = init_result.solution

        result = root_solver(
            kappa,
            gamma,
            alpha,
            warm_start,
            main_method,
            gh,
            prox_tol,
            transform,
            **root_kwargs,
        )

    return result
