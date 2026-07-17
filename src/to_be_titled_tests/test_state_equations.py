import warnings

import numpy as np
import pytest
from scipy.special import expit, roots_hermite

from to_be_titled import state_equations

# TODO: doctrings explaining all tests


def test_prox_inverse_identity() -> None:
    b = 2.5

    # Generate an array of known 'u' values
    true_u = np.linspace(-10, 10, 1000)

    # Calculate the corresponding 'x' inputs
    x_input = b * expit(true_u) + true_u

    estimated_u = state_equations._proximal_operator(x_input, b)

    np.testing.assert_allclose(true_u, estimated_u, atol=1e-9)


@pytest.mark.parametrize("b", [40.0, 15.0, 1.0, 0.1])
def test_prox_zero_point(b: float) -> None:
    # If x = b / 2 then the true root 'u' must be exactly 0
    x_input = b / 2

    estimated_u = state_equations._proximal_operator(x_input, b)

    np.testing.assert_allclose(estimated_u, 0.0, atol=1e-9)


def test_prox_asymptotics() -> None:
    b = 5.0

    # For large positive u values; expit(u) tends to 1,
    # u \approx x - b
    x_pos = np.array([500.0, 1000.0])
    u_pos = state_equations._proximal_operator(x_pos, b)

    np.testing.assert_allclose(u_pos, x_pos - b, rtol=1e-5)

    # For large negative u values; expit(u) tends to 0
    # u approx x
    x_neg = np.array([-500.0, -1000.0])
    u_neg = state_equations._proximal_operator(x_neg, b)
    np.testing.assert_allclose(u_neg, x_neg, rtol=1e-5)


def test_se0_shape_and_reproducibility() -> None:
    mu, b, sigma, kappa, gamma = 0.5, 1.0, 1.0, 0.4, 2.0
    alpha = 1 / (1 + kappa)

    res1 = state_equations._se_no_intercept(mu, b, sigma, kappa, gamma, alpha)
    res2 = state_equations._se_no_intercept(mu, b, sigma, kappa, gamma, alpha)

    assert isinstance(res1, np.ndarray)
    assert res1.shape == (3,)

    np.testing.assert_array_equal(res1, res2)


def test_se0_gh_reproducibility() -> None:
    mu, b, sigma, kappa, gamma = 0.5, 1.0, 1.0, 0.4, 2.0
    alpha = 1 / (1 + kappa)

    res_gh_internal = state_equations._se_no_intercept(
        mu, b, sigma, kappa, gamma, alpha, gh=None
    )

    gh_precomputed = roots_hermite(200)
    res_gh_external = state_equations._se_no_intercept(
        mu, b, sigma, kappa, gamma, alpha, gh=gh_precomputed
    )

    np.testing.assert_allclose(res_gh_internal, res_gh_external, atol=1e-12)


# TODO: more rigorously test extreme parameter values
def test_se0_no_nan_or_inf() -> None:
    # Test extreme kappa gamma pairs which push mu to zero
    mu_tiny = 1e-100
    b, sigma, kappa, gamma, alpha = 50.0, 5.0, 0.9, 15.0, 1 / (1 + 0.9)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = state_equations._se_no_intercept(
                mu_tiny, b, sigma, kappa, gamma, alpha
            )
            assert not np.any(np.isnan(res))
        except RuntimeWarning as e:
            pytest.fail(f"Mathematical instability detected: {e}")


# TODO: test for cancellation precision when (kappa**2 * sigma**2) - b**2
# are large and similar in magnitude

# TODO: test se0 evaluates similarly across difference prox_tols

# TODO : add exact values obtained from brglm2 se0 and compare
