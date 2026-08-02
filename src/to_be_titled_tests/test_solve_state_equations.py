import numpy as np
import pytest

from to_be_titled.solvers import solve_state_equation
from to_be_titled.types import FloatArray


# Compare solve_se to Candes and Sur results.
# See Table 13, set alpha = 1, when gamma = np.sqrt(5 + beta0^2)
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
    kappa = 0.2
    alpha = 1.0
    gamma = np.sqrt(5 + thetas**2)

    # Use default brglm2 starting guess
    start = np.array([0.5, 1, 1])

    solver_result, _ = solve_state_equation(kappa, gamma, alpha, start, init_iter=50)

    # Ensure solver converged successfully
    assert solver_result.success is True, "Solver failed to converge"
    np.testing.assert_allclose(solver_result.solution.to_array(), roots, atol=1e-2)


# Compare with Candes Table 13, with intercept
# alpha = 1.0,
@pytest.mark.parametrize(
    "thetas, roots",
    [
        (0.0, np.array([1.50, 3.03, 4.7, 0.0])),
        (0.5, np.array([1.51, 3.13, 4.84, 0.76])),
        (1.0, np.array([1.56, 3.45, 5.16, 1.559])),
        (2, np.array([1.83, 5.47, 7.01, 3.68])),
        (2.5, np.array([2.31, 8.96, 10.0, 5.8])),
    ],
)
def test_solve_state_equation_w_int_compare_candes_sur(
    thetas: float, roots: FloatArray
) -> None:
    gamma = np.sqrt(5)
    kappa = 0.2
    alpha = 1.0

    start = np.array([0.5, 1, 1, 0.0])

    solver_result, _ = solve_state_equation(
        kappa,
        gamma,
        alpha,
        start,
        intercept=thetas,
        init_iter=50,
    )

    # Ensure solver converged successfully
    assert solver_result.success is True, "Solver failed to converge"
    np.testing.assert_allclose(solver_result.solution.to_array(), roots, atol=1e-1)


# Test against solve_state_equations w/o intecept against brglm2
# for different kappa/gamma/alpha
def test_solve_state_equations_against_se0_brglm2() -> None:
    """
    Given a known true signal strength, this tests that the solver
    is able to find `mu`, `b`, `sigma` such that the 3 state equations
    evaluate to zero. Moreover, this test ensures roots are suffciently
    close to values found brglm2::solve_se().
    """
    kappa, gamma, alpha = 0.2, 5, 0.88

    # results from brglm2::solve_se
    true_nelder_mead_50 = np.array([0.5649718, 2.5935673, 2.5376760])

    # use default, naive guess used in brglm2
    start = np.array([0.5, 1, 1])

    est_nelder_mead_50, _ = solve_state_equation(
        kappa, gamma, alpha, start, init_iter=50
    )

    # Ensure solver gets suffciently to roots
    np.testing.assert_array_almost_equal(est_nelder_mead_50.func_value, np.zeros(3))

    # Ensure solver roots match brglm2 roots
    np.testing.assert_allclose(
        est_nelder_mead_50.solution.to_array(), true_nelder_mead_50, atol=1e-7
    )


# Test against solve_state_equations with intercept against brglm2
# for different kappa/gamma/alpha
def test_solve_state_equations_not_corrupt_against_se1_brglm2() -> None:
    """
    This test asserts that the solver with corrupted = False, obtains the `mu`,
    `b`, `sigma` and `iota` such that the 4 state equations evaluate approximately
    close to zero. Moreover, this test asserts that the found roots, match the
    roots found using brglm2::solve_se().
    """
    kappa, gamma, alpha, theta = 0.2, 5, 0.88, 1.0
    # results from brglm2::solve_se using above values
    brglm2_res = np.array([0.5565527, 2.6079197, 2.5297002, 0.5585552])

    start = np.array([0.5, 1, 1, 0])

    soln, _ = solve_state_equation(
        kappa,
        gamma,
        alpha,
        start,
        corrupted=False,
        intercept=theta,
    )

    # checks solver gets approximately close to roots
    np.testing.assert_array_almost_equal(soln.func_value, np.zeros(4))
    # check solver roots match brglm2
    np.testing.assert_allclose(soln.solution.to_array(), brglm2_res, atol=1e-10)


def test_solve_state_equations_corrupt_against_se1_brglm2() -> None:
    """
    The test uses brglm2::solve_se to obtain `mu`,`b`,
    `sigma` and `iota` when corrupted is False, and to compute corrupted
    signal strength `nu`. These values are used to evaluate solver when
    corrupted = True, and intercept is found `iota` from brglm2. Solver's
    solution is compared against brglm2::solve_se() results.
    """
    kappa, gamma, alpha, theta = 0.2, 5, 0.88, 1.0

    # results from brglm2::solve_se using above values
    brglm2_res = np.array([0.5565527, 2.6079197, 2.5297002, 0.5585552])
    mu_root, b_root, sigma_root, iota_root = brglm2_res

    # compute the corrupted signal strength
    nu = np.sqrt(mu_root**2 * gamma**2 + kappa * sigma_root**2)

    start = np.array([0.5, 1, 1, 0])

    # pass in corrupted signal strength and estimtaed intercept (iota_root)
    soln_c, _ = solve_state_equation(
        kappa,
        nu,
        alpha,
        start,
        corrupted=True,
        intercept=iota_root,
    )

    # 4. The expected corrupted result: mu, b, sigma, and the TRUE intercept (theta)
    brglm2_corrupted_res = np.array([mu_root, b_root, sigma_root, theta])

    np.testing.assert_array_almost_equal(soln_c.func_value, np.zeros(4))
    np.testing.assert_allclose(
        soln_c.solution.to_array(), brglm2_corrupted_res, atol=1e-6
    )


def test_solve_se0_with_nu() -> None:
    """
    Tests that when using `mu`, `b`, `sigma` found from the solver to compute the
    corrupted signal strength `nu`, the corrupted solver with nu, returns
    the same solution as the uncorrupted solver.
    """
    kappa, gamma, alpha = 0.2, 5, 0.88

    start = np.array([0.5, 1, 1])

    res0, _ = solve_state_equation(kappa, gamma, alpha, start)
    sol0 = res0.solution.to_array()
    (
        mu,
        _,
        sigma,
    ) = sol0[0], sol0[1], sol0[2]

    # Compute the corrupted signal strength
    nu = np.sqrt(mu**2 * gamma**2 + kappa * sigma**2)

    # Use the corrupted signal strength as gamma in solver with corrupted = True
    sol0_c, _ = solve_state_equation(kappa, nu, alpha, start, corrupted=True)

    np.testing.assert_array_almost_equal(sol0_c.solution.to_array(), sol0, decimal=8)


def test_solve_se1_retrieve_nu() -> None:
    """
    Test that the solver with intercept can retrieve same `mu`, `b`, `sigma` with
    corrupted and true signal strength. Ensures corrupted solver recovers
    true intercept (`theta`).
    """
    kappa, gamma, alpha, theta = 0.2, 5, 0.88, 1.0

    start = np.array([0.5, 1, 1, 0.0])

    res1, _ = solve_state_equation(kappa, gamma, alpha, start, intercept=theta)
    sol1 = res1.solution.to_array()

    mu, _, sigma, iota = (sol1[0], sol1[1], sol1[2], sol1[3])

    # Compute the corrupted signal strength
    nu = np.sqrt(mu**2 * gamma**2 + kappa * sigma**2)

    # Use the corrupted signal strength as gamma and iota as intercept
    # in solver with corrupted = True
    res1_c, _ = solve_state_equation(
        kappa, nu, alpha, start, corrupted=True, intercept=iota
    )
    sol1_c = res1_c.solution.to_array()

    # mu, b, sigma should be recovered consistently between the two parameterizations
    np.testing.assert_array_almost_equal(sol1_c[:3], sol1[:3], decimal=8)
    # the corrupted-branch free parameter should recover the TRUE intercept (theta),
    np.testing.assert_almost_equal(sol1_c[3], theta, decimal=8)


def test_solve_state_equation_no_int_transform_safely() -> None:
    """
    Ensures solver without intercept, safely log transforms the
    parameter
    """
    kappa, gamma, alpha = 0.2, 5, 0.88

    start = np.asarray([0.5, 1, 1])

    soln, _ = solve_state_equation(kappa, gamma, alpha, start, transform=False)
    soln_t, _ = solve_state_equation(kappa, gamma, alpha, start, transform=True)

    np.testing.assert_array_almost_equal(
        soln.solution.to_array(), soln_t.solution.to_array()
    )


def test_solve_state_equation_int_transform_safely() -> None:
    """
    Ensures solver with intercept, safely log transform the
    parameter
    """
    kappa, gamma, alpha, theta = 0.2, 5, 0.88, 1.0

    start = np.asarray([0.5, 1, 1, 0])

    soln, _ = solve_state_equation(
        kappa, gamma, alpha, start, transform=False, intercept=theta
    )
    soln_t, _ = solve_state_equation(
        kappa,
        gamma,
        alpha,
        start,
        transform=True,
        intercept=theta,
    )

    np.testing.assert_array_almost_equal(
        soln.solution.to_array(), soln_t.solution.to_array()
    )
