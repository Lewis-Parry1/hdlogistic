import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import expit

from to_be_titled.state_equations import (
    _proximal_operator,  # pyright: ignore[reportPrivateUsage]
    se_no_intercept,
    se_with_intercept,
)

"""
These tests test the proximal operator to ensure that Newton's method
converges to expected values even when the magnitude of the input is large/small
and ensure the system's state equations are evaluated as expected.

The functions will be tested on a range of plausible state equation input values
for `mu`, `b`, `sigma` as well as for various `kappa` (p/n) and `gamma`
(signal strength) ranges. `alpha` will defined as 1 / (1 + `kappa`) to
discussed in Sterzinger and Kosmidis (2026) to shrink as the proprotion of
covariates to observations grows. This is known as 'adaptive shrinkage'.

We define the following regimes for params = (mu, b, sigma) with
the (kappa, gamma) pair.
1. Low dimensional / unbiased reigmes : mu -> 1, kappa -> 0 , gamma = sqrt(0.9),
mu = (0.9, 0.95, 0.99), b = 1.0, sigma = 1.0
1. Phase transition boundary: (kappa, gamma) = [(0.125, 12.5), (0.25, 5),
(0.5, 2.5)] with mu = 0.4 (see Fig 3. [1]) b = 2.0, sigma = 2.0.
2. High dimensional / extreme shrinkage reigme: mu -> 0, so test
mu = (1e-2, 1e-4, 1e-8, 1e-20),  b = 50, sigma = 5, with kappa = 0.9 and gamma = 15.

References:
.. [1] P.Sterzinger and I.Kosmidis, 2026.
"""


# Test 1 & 2 : Ensure proximal operator successfully converges
# suffciently close to known true values
@pytest.mark.parametrize("b", [0.1, 1.0, 15.0, 40.0, 100.0])
def test_prox_inverse_identity(b: float) -> None:
    """
    Given a set of known true `u` values, differentiating the convex function
    f(x,b) = b*ln(1+e^u)+1/2 * ​(x−u)^2 with respect to u, yields
    f'(x,b) = b * expit(u) + u - x. Setting f'(x,b) = 0, means
    x = b * expit (u) + u.

    True `u` values are generated, and the corresponding `x` values are computed.
    This test assess whether the proximal operator estimates u suffciently
    close to the true u being defined.
    """

    true_u = np.linspace(-20, 20, 1000)

    x_input = b * expit(true_u) + true_u

    u_est = _proximal_operator(x_input, b)

    np.testing.assert_allclose(true_u, u_est, atol=1e-9)


@pytest.mark.parametrize("b", [0.1, 1.0, 15.0, 40.0, 100.0])
def test_prox_zero_point(b: float) -> None:
    """
    Evaluates the known analytic root where x = b / 2 yields exactly u = 0.
    """
    x_input = b / 2.0

    u_est = _proximal_operator(x_input, b)

    np.testing.assert_allclose(u_est, 0.0, atol=1e-9)


# Define data reigmes
# (mu, b, sigma, kappa, gamma)
REGIMES: list[tuple[float, float, float, float, float]] = [
    # 1. Low dimensional / unbiased (mu -> 1, kappa -> 0, gamma = sqrt(0.9))
    *((mu, 1.0, 1.0, 0.1, np.sqrt(0.9)) for mu in [0.9, 0.95, 0.99]),
    # 2. Phase transition boundary (mu = 0.4, b = 2.0, sigma = 2.0)
    *(
        (0.4, 2.0, 2.0, kappa, gamma)
        for kappa, gamma in [(0.125, 12.5), (0.25, 5.0), (0.5, 2.5)]
    ),
    # 3. High dimensional / extreme shrinkage
    # (kappa = 0.9, gamma = 15, b = 50, sigma = 5)
    *((mu, 50.0, 5.0, 0.9, 15.0) for mu in [1e-2, 1e-4, 1e-8, 1e-20]),
]


# Test 3: Test _proximal_operator and se_no_intercept on different parameter reigmes
@pytest.mark.parametrize("mu, b, sigma, kappa, gamma", REGIMES)
def test_prox_data_regimes(
    mu: float, b: float, sigma: float, kappa: float, gamma: float
) -> None:
    """
    Ensures that prox_operator converges for specified reigmes when given
    the exact arrays the proximal operator will recieve inside se_no_intercept.
    """
    # Simulate extreme grid points to simulate the edges of the Gauss-Hermite grid
    # (The roots of a 200-node Hermite polynomial span roughly -19.3 to 19.3)
    x_grid = np.array([-19.3, 0.0, 19.3])
    y_grid = np.array([-19.3, 0.0, 19.3])
    x, y = np.meshgrid(x_grid, y_grid)

    alpha = 1.0 / (1.0 + kappa)
    a_frac = 0.5 * (1 + alpha)

    # Compute the exact input to prox
    q1 = np.sqrt(2) * gamma * x.flatten()
    q2 = (q1 * mu) + np.sqrt(2) * (np.sqrt(kappa) * sigma * y.flatten())
    x_input = q2 + a_frac * b

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        u_est = _proximal_operator(x_input, b)

        assert not np.any(np.isnan(u_est))

        u_est = np.asarray(_proximal_operator(x_input, b))
        assert u_est.shape == x_input.shape


@pytest.mark.parametrize("mu, b, sigma, kappa, gamma", REGIMES)
def test_se_no_intercept_regimes(
    mu: float, b: float, sigma: float, kappa: float, gamma: float
) -> None:
    """
    Evaluates the full state equations on entire Gauss-Hermite grid
    for the specified data regimes to guarantee integration stability.
    """
    alpha = 1.0 / (1.0 + kappa)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = se_no_intercept(mu, b, sigma, kappa, gamma, alpha)

            # Guarantee the equations successfully returned 3 valid floating points
            assert not np.any(np.isnan(res))
            assert len(res) == 3

        except RuntimeWarning as e:
            pytest.fail(
                f"Math instability at mu={mu}, kappa={kappa}, gamma={gamma}: {e}"
            )


def test_se_no_intercept_shape_and_reproducibility() -> None:
    """
    Gauss Hermite nodes and weights are generated within _se_no_intercept.
    This test checks that the array returns 3 values and that _se_no_intercept
    is deterministic.
    """
    # standard parameters
    mu, b, sigma, kappa, gamma = 0.5, 1.0, 1.0, 0.2, np.sqrt(0.9)
    alpha = 1 / (1 + kappa)

    res1 = se_no_intercept(mu, b, sigma, kappa, gamma, alpha)
    res2 = se_no_intercept(mu, b, sigma, kappa, gamma, alpha)

    assert isinstance(res1, np.ndarray)
    assert res1.shape == (3,)

    np.testing.assert_array_equal(res1, res2)


@pytest.mark.parametrize(
    "mu, b, sigma, kappa, gamma, alpha, expected_res",
    [
        # 1. Low Dimensional / Unbiased Regime
        (
            0.95,
            1.0,
            1.0,
            0.1,
            np.sqrt(0.9),
            1.0 / (1.0 + 0.1),
            np.array([-0.004859819, 0.069734687, -0.108815242]),
        ),
        # 2. Phase Transition Boundary
        (
            0.4,
            2.0,
            2.0,
            0.5,
            2.5,
            1.0 / (1.0 + 0.5),
            np.array([0.07747317, -0.24810826, 0.73448663]),
        ),
        # 3. High Dimensional / Extreme Shrinkage Regime
        (
            1e-20,
            50.0,
            5.0,
            0.9,
            15.0,
            1.0 / (1.0 + 0.9),
            np.array([0.263890826, -0.003967611, 0.940855077]),
        ),
    ],
)
def test_se_no_intercept_matches_brglm2_se0(
    mu: float,
    b: float,
    sigma: float,
    kappa: float,
    gamma: float,
    alpha: float,
    expected_res: NDArray[np.float64],
) -> None:
    """
    Test to ensure se_no_intercept matches the equivalent brglm2 se0. Both functions
    approximate the value of the three state equations.
    """
    # Evaluate the state equations in Python
    res = se_no_intercept(
        mu=mu, b=b, sigma=sigma, kappa=kappa, gamma=gamma, alpha=alpha
    )

    np.testing.assert_allclose(res, expected_res, atol=1e-7)


def test_se_with_intercept_matches_brglm2_se1() -> None:
    kappa0 = 0.2
    gamma0 = 5
    alpha0 = 0.88
    theta0 = 1
    iota0 = 2
    mu0 = 0.7
    b0 = 1.2
    sigma0 = 2.3

    soln = se_with_intercept(mu0, b0, sigma0, iota0, kappa0, gamma0, alpha0, theta0)

    brglm_results = np.asarray([-0.05090216, -0.11007367, 0.11183220, -0.10479934])
    np.testing.assert_allclose(brglm_results, soln, atol=1e-7)


def test_se0_se1_is_equal() -> None:
    """
    That the 4-parameter system with an intercept matches the
    3-parameter system when the intercept terms is nullified.
    """
    kappa, gamma, alpha = 0.2, 5, 0.88
    mu, b, sigma = 0.7, 1.2, 2.3

    sol0 = se_no_intercept(mu, b, sigma, kappa, gamma, alpha)
    sol1 = se_with_intercept(mu, b, sigma, 0, kappa, gamma, alpha, intercept=0)
    # Assert, almost equal, a inisgniciant numerical differences
    # occur
    np.testing.assert_array_almost_equal(sol0, sol1[0:3])
