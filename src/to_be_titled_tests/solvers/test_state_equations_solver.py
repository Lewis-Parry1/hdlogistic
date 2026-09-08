from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from to_be_titled.inference import derive_nu_from_gamma
from to_be_titled.solvers.state_equations_solver import (
    SolverConvergenceWarn,
    solve_state_equation,
)
from to_be_titled.types import FloatArray


@pytest.mark.candes_sur
@pytest.mark.parametrize(
    "thetas, roots",
    [
        (0.0, np.array([1.50, 3.03, 4.75])),
        (0.5, np.array([1.51, 3.12, 4.84])),
        (1.0, np.array([1.55, 3.42, 5.13])),
        (2, np.array([1.75, 4.83, 6.45])),
        (2.5, np.array([1.95, 6.26, 7.73])),
    ],
)
def test_solve_state_equation_no_int_compare_candes_sur(
    thetas: float, roots: FloatArray
) -> None:
    """Test to compare solver without intercept solution against known true
    values from Candes & Sur Table 13."""
    kappa = 0.2
    alpha = 1.0
    gamma = np.sqrt(5 + thetas**2)

    solver_result, chain = solve_state_equation(
        kappa,
        gamma,
        alpha,
        warn_interpolator_issues=False,
    )

    print(chain)

    assert solver_result.success is True, "Solver failed to converge"
    np.testing.assert_allclose(solver_result.solution.to_array(), roots, atol=1e-2)


# Compare with Candes Table 13, with intercept
# alpha = 1.0,
@pytest.mark.candes_sur
@pytest.mark.parametrize(
    "thetas, roots",
    [
        (0.5, np.array([1.51, 3.13, 4.84, 0.76])),
        (1.0, np.array([1.56, 3.45, 5.16, 1.559])),
        (2, np.array([1.83, 5.47, 7.01, 3.68])),
        (2.5, np.array([2.31, 8.96, 10.0, 5.8])),
    ],
)
def test_solve_state_equation_w_int_compare_candes_sur(
    thetas: float, roots: FloatArray
) -> None:
    """Test to compare solver with intercept solution against known true
    values from Candes & Sur Table 13."""
    gamma = np.sqrt(5)
    kappa = 0.2
    alpha = 1.0

    solver_result, chain = solve_state_equation(
        kappa,
        gamma,
        alpha,
        intercept=thetas,
        warn_interpolator_issues=False,
    )

    print(chain)

    np.testing.assert_allclose(solver_result.solution.to_array(), roots, atol=1e-1)


@pytest.mark.brglm2
def test_solve_state_equations_against_se0_brglm2() -> None:
    """
    Given a known true signal strength (gamma), this tests that the solver
    is able to find `mu`, `b`, `sigma` such that the 3 state equations
    evaluate to zero. Moreover, this test ensures roots are suffciently
    close to values found brglm2::solve_se().
    """
    kappa, gamma, alpha = 0.2, 5, 0.88

    res_brglm2 = np.array([0.5649718, 2.5935673, 2.5376760])

    res, chain = solve_state_equation(
        kappa, gamma, alpha, warn_interpolator_issues=False
    )

    print(chain)

    np.testing.assert_array_almost_equal(res.func_value, np.zeros(3))

    np.testing.assert_allclose(res.solution.to_array(), res_brglm2, atol=1e-7)


@pytest.mark.brglm2
def test_solve_state_equations_not_corrupt_against_se1_brglm2() -> None:
    """
    This test asserts that in the uncorrupted case, the solver obtains the `mu`,
    `b`, `sigma` and `iota` such that the 4 state equations evaluate approximately
    close to zero. Moreover, this test asserts that the found roots, match the
    roots found using brglm2::solve_se().
    """
    kappa, gamma, alpha, theta = 0.2, 5, 0.88, 1.0
    # results from brglm2::solve_se using above values
    brglm2_res = np.array([0.5565527, 2.6079197, 2.5297002, 0.5585552])

    soln, chain = solve_state_equation(
        kappa,
        gamma,
        alpha,
        corrupted=False,
        intercept=theta,
        warn_interpolator_issues=False,
    )

    print(chain)

    # checks solver gets approximately close to roots
    np.testing.assert_array_almost_equal(soln.func_value, np.zeros(4))
    # check solver roots match brglm2
    np.testing.assert_allclose(soln.solution.to_array(), brglm2_res, atol=1e-10)


@pytest.mark.brglm2
def test_solve_state_equations_corrupt_against_se1_brglm2() -> None:
    """
    Defining `kappa`, `gamma`, `alpha` and `theta`, this test
    asserts that the solution from the uncorrupted brglm2 solver
    is suffciently close with the corresponding corrupted solver.
    """
    kappa, gamma, alpha, theta = 0.2, 5, 0.88, 1.0

    # results from brglm2::solve_se using above values
    brglm2_res = np.array([0.5565527, 2.6079197, 2.5297002, 0.5585552])
    mu, b, sigma, iota = brglm2_res  # approximate roots

    nu = derive_nu_from_gamma(kappa, gamma, mu, sigma)

    # pass in corrupted signal strength and estimtaed intercept (iota)
    soln_corrupt, chain = solve_state_equation(
        kappa,
        nu,
        alpha,
        corrupted=True,
        intercept=iota,
        warn_interpolator_issues=False,
    )

    print(chain)

    # the expected corrupted result from brglm2: mu, b, sigma,
    # and the TRUE intercept (theta)
    brglm2_corrupted_res = np.array([mu, b, sigma, theta])

    np.testing.assert_array_almost_equal(soln_corrupt.func_value, np.zeros(4))
    np.testing.assert_allclose(
        soln_corrupt.solution.to_array(), brglm2_corrupted_res, atol=1e-6
    )


def test_solve_se0_with_nu() -> None:
    """
    Tests that when using `mu`, `b`, `sigma` found from the solver to compute the
    corrupted signal strength `nu`, the corrupted solver with nu, returns
    the same solution as the uncorrupted solver.
    """
    kappa, gamma, alpha = 0.2, 5, 0.88

    res0, chain = solve_state_equation(
        kappa, gamma, alpha, warn_interpolator_issues=False
    )

    print(chain)

    nu = derive_nu_from_gamma(kappa, gamma, res0.solution.mu, res0.solution.sigma)

    res0_c, chain_c = solve_state_equation(
        kappa, nu, alpha, corrupted=True, warn_interpolator_issues=False
    )

    print(chain_c)

    np.testing.assert_array_almost_equal(
        res0_c.solution.to_array(), res0.solution.to_array(), decimal=8
    )


def test_solve_se1_retrieve_nu() -> None:
    """
    Test that the solver with intercept can retrieve same `mu`, `b`, `sigma` with
    corrupted and true signal strength. Ensures corrupted solver recovers
    true intercept (`theta`).
    """
    kappa, gamma, alpha, theta = 0.2, 5, 0.88, 1.0

    res1, chain = solve_state_equation(
        kappa, gamma, alpha, intercept=theta, warn_interpolator_issues=False
    )

    print(chain)

    # Compute the corrupted signal strength
    nu = derive_nu_from_gamma(kappa, gamma, res1.solution.mu, res1.solution.sigma)

    # Use the corrupted signal strength as gamma and iota as intercept
    # in solver with corrupted = True
    res1_c, chain_c = solve_state_equation(
        kappa,
        nu,
        alpha,
        corrupted=True,
        intercept=res1.solution.iota,
        warn_interpolator_issues=False,
    )

    print(chain_c)

    # mu, b, sigma should be recovered consistently between the two parameterizations
    np.testing.assert_array_almost_equal(
        res1_c.solution.to_array()[:3], res1.solution.to_array()[:3], decimal=8
    )
    # the corrupted-branch free parameter should recover the TRUE intercept (theta),
    np.testing.assert_almost_equal(res1_c.solution.to_array()[3], theta, decimal=8)


def test_solve_state_equation_no_int_transform_safely() -> None:
    """
    Ensures solver without intercept, safely log transforms the
    parameter
    """
    kappa, gamma, alpha = 0.2, 5, 0.88

    soln, _ = solve_state_equation(
        kappa, gamma, alpha, transform=False, warn_interpolator_issues=False
    )
    soln_t, _ = solve_state_equation(
        kappa, gamma, alpha, transform=True, warn_interpolator_issues=False
    )

    np.testing.assert_array_almost_equal(
        soln.solution.to_array(), soln_t.solution.to_array()
    )


def test_solve_state_equation_int_transform_safely() -> None:
    """
    Ensures solver with intercept, safely log transform the
    parameter
    """
    kappa, gamma, alpha, theta = 0.2, 5, 0.88, 1.0

    soln, _ = solve_state_equation(
        kappa,
        gamma,
        alpha,
        transform=False,
        intercept=theta,
        warn_interpolator_issues=False,
    )
    soln_t, _ = solve_state_equation(
        kappa,
        gamma,
        alpha,
        transform=True,
        intercept=theta,
        warn_interpolator_issues=False,
    )

    np.testing.assert_array_almost_equal(
        soln.solution.to_array(), soln_t.solution.to_array()
    )


@pytest.mark.brglm2
def test_rigon_alverti_case() -> None:
    kappa = 0.5
    gamma = np.sqrt(5)
    alpha = 1 / (1 + kappa)

    res, chain = solve_state_equation(
        kappa, gamma, alpha, corrupted=False, warn_interpolator_issues=False
    )
    print(chain)

    brglm2_res = np.array([0.5095007, 6.3607799, 1.9872668])

    np.testing.assert_allclose(res.solution.to_array(), brglm2_res, atol=1e-2)


KAPPAS = np.linspace(0.05, 0.95, 20)
GAMMAS = np.linspace(0.5, 20, 20)
GRID_POINTS = list(product(KAPPAS, GAMMAS))


@pytest.mark.slow
@pytest.mark.parametrize("kappa, gamma", GRID_POINTS)
def test_grid_sweep_solver(kappa: float, gamma: float) -> None:
    alpha = 1 / (1 + kappa)

    try:
        result, _ = solve_state_equation(
            kappa, gamma, alpha, start=None, warn_interpolator_issues=False
        )
    except SolverConvergenceWarn as exc:
        pytest.fail(f"Solver failed to converge at kappa={kappa}, gamma={gamma}: {exc}")

    np.testing.assert_allclose(
        result.func_value, np.zeros_like(result.func_value), atol=1e-8, rtol=1e-6
    )
