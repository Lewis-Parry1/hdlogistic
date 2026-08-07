"""
This script generates a 100x100 grid of true values for the parameters
`mu`, `b` and `sigma` given a 1D arrays of `kappa` and `gamma` points of
length 100. A continuation strategy is used, wherein the closest previous
solution is fed in to _root_solver() directly using 'hybr' method. This
is sufficient to quickly recover the approximate true roots of the
MDYPL state equation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from to_be_titled.solvers.solver_types import SolverResult
from to_be_titled.solvers.state_equations_solver import (
    _default_start,
    _init_solver,
    _root_solver,
)
from to_be_titled.types import FloatArray
from to_be_titled.validation import _is_valid_domain

OUTPUT_PATH = Path(__file__).resolve().parent / "true_reference_param_grid.npz"


def _solver_strategy(
    kappa: float, gamma: float, alpha: float, start: FloatArray
) -> SolverResult | None:

    solver_result = _root_solver(kappa, gamma, alpha, start)
    if solver_result.success and _is_valid_domain(solver_result.solution.to_array()):
        return solver_result

    default_guess = _default_start(False)
    solver_result = _root_solver(kappa, gamma, alpha, default_guess)
    if solver_result.success and _is_valid_domain(solver_result.solution.to_array()):
        return solver_result

    init_result = _init_solver(kappa, gamma, alpha, default_guess, "BFGS")
    warm_start = init_result.solution.to_array()
    solver_result = _root_solver(kappa, gamma, alpha, warm_start)
    if solver_result.success and _is_valid_domain(solver_result.solution.to_array()):
        return solver_result
    else:
        print(f"Failed at kappa = {kappa} and gamma = {gamma}\n")
        return None


def evaluate_grid(
    n_points_axis: int,
    kappa_range: tuple[float, float],
    gamma_range: tuple[float, float],
) -> dict[str, FloatArray]:

    kappa_arr = np.linspace(kappa_range[0], kappa_range[1], num=n_points_axis)
    gamma_arr = np.linspace(gamma_range[0], gamma_range[1], num=n_points_axis)

    mu_grid = np.full((n_points_axis, n_points_axis), np.nan, dtype=np.float64)
    b_grid = np.full((n_points_axis, n_points_axis), np.nan, dtype=np.float64)
    sigma_grid = np.full((n_points_axis, n_points_axis), np.nan, dtype=np.float64)

    previous_solution: FloatArray = _default_start(has_intercept=False)

    for i, gamma in enumerate(gamma_arr):
        # continuation startegy, left to right sweep
        kappa_iter = (
            range(n_points_axis) if i % 2 == 0 else range(n_points_axis - 1, -1, -1)
        )

        for j in kappa_iter:
            kappa = kappa_arr[j]
            alpha = 1 / (1 + kappa)  # use adaptive shrinkage

            result = _solver_strategy(kappa, gamma, alpha, previous_solution)

            if result is not None:
                soln = result.solution
                # index such that we preserve (kappa, gamma) ordering
                mu_grid[j, i] = soln.mu
                b_grid[j, i] = soln.b
                sigma_grid[j, i] = soln.sigma
                previous_solution = soln.to_array()

    return {
        "kappa": kappa_arr,
        "gamma": gamma_arr,
        "mu": mu_grid,
        "b": b_grid,
        "sigma": sigma_grid,
    }


if __name__ == "__main__":
    grid = evaluate_grid(
        n_points_axis=100, kappa_range=(0.01, 0.975), gamma_range=(0.1, 25)
    )
    np.savez_compressed(
        OUTPUT_PATH,
        **grid,
    )
