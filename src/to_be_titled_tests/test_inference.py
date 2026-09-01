import numpy as np
from scipy.special import expit

from to_be_titled.inference import (
    _generalised_binomial_pmf,
    compute_sloe,
    compute_taus,
    derive_gamma_from_nu,
    derive_nu_from_gamma,
)

# ---- SLOE Tests ----


def test_compute_sloe_estimator_no_leverage_adjustment() -> None:
    """If h_i = 0 for all observations, the sloe scores collapse
    to being just the linear_predictors.
    """
    linear_predictors = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    y_adjusted = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    leverages = np.zeros(3, dtype=np.float64)
    fitted_probs = expit(linear_predictors)

    expected = float(np.std(linear_predictors, ddof=1))
    np.testing.assert_allclose(
        compute_sloe(y_adjusted, linear_predictors, fitted_probs, leverages),
        expected,
    )


def test_compute_sloe_estimator_excludes_infinite_values() -> None:
    """If h_i = 1.0 then a divide by zero error occurs. These values
    are ignored. For the other zero leverage values, the expected value
    is the std of the linear predictors.
    """
    linear_predictors = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    y_adjusted = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    leverages = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    fitted_probs = expit(linear_predictors)

    expected = float(np.std(np.array([0.0, -1.0], dtype=np.float64), ddof=1))
    np.testing.assert_allclose(
        compute_sloe(y_adjusted, linear_predictors, fitted_probs, leverages),
        expected,
    )


def test_compute_sloe_estimator_happy_path() -> None:
    """Verify SLOE calculation with standard inputs."""

    # Setup inputs where 0 < mus < 1 and 0 < leverages < 1
    linear_predictors = np.array([0.0, 0.5, -0.5], dtype=np.float64)
    y_adjusted = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    leverages = np.array([0.2, 0.3, 0.1], dtype=np.float64)
    fitted_probs = expit(linear_predictors)

    # Expected sample standard deviation of S with ddof=1
    expected_signal_strength = 1.3258861214054045

    np.testing.assert_allclose(
        compute_sloe(y_adjusted, linear_predictors, fitted_probs, leverages),
        expected_signal_strength,
        atol=1e-4,
    )


## --- Test gamma to nu functions ---
def test_derive_nu_from_gamma_expected():
    mu, gamma, kappa, sigma = 0.5, 1, 0.3, 1
    expected_nu = float(np.sqrt(55) / 10)

    nu = derive_nu_from_gamma(kappa, gamma, mu, sigma)

    np.testing.assert_almost_equal(expected_nu, nu)


def test_derive_gamma_from_nu_expected():
    kappa, nu, sigma, mu = 0.3, 1, 1, 0.5
    expected_gamma = float(np.sqrt(70) / 5)

    gamma = derive_gamma_from_nu(kappa, nu, sigma, mu)
    np.testing.assert_almost_equal(expected_gamma, gamma)


# --- Test _dy_binomial_coeffcient ---


def test_generalised_binomial_pmf_expected():
    y_adj = np.asarray([0.5, 0.5, 0.2])
    freq_weights = np.asarray([1.0, 1.0, 2.0])
    fitted_probs = np.asarray([0.8, 0.8, 0.5])

    pmf = _generalised_binomial_pmf(y_adj, freq_weights, fitted_probs, log=False)

    expected = np.asarray([0.5092958179, 0.5092958179, 0.3941805878])
    np.testing.assert_allclose(expected, pmf)


def test_generalised_binomial_pmf_log_expected():
    y_adj = np.asarray([0.5, 0.5, 0.2])
    freq_weights = np.asarray([1.0, 1.0, 2.0])
    fitted_probs = np.asarray([0.8, 0.8, 0.5])

    pmf = _generalised_binomial_pmf(y_adj, freq_weights, fitted_probs, log=True)

    expected = np.asarray(np.log([0.5092958179, 0.5092958179, 0.3941805878]))
    np.testing.assert_allclose(expected, pmf)


## --- Test compute taus ----


def test_compute_taus_with_intercept():
    x = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    intercept_index = 0

    expected = np.asarray([1.154701, 1.154701])

    np.testing.assert_allclose(compute_taus(x, intercept_index), expected, rtol=1e-6)


def test_compute_taus_with_no_intercept():
    x = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    intercept_index = None

    expected = np.asarray([1.154701, 1.154701])

    np.testing.assert_allclose(compute_taus(x, intercept_index), expected, rtol=1e-6)
