import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from to_be_titled.inference import (
    compute_likelihood,
    compute_sloe,
    compute_taus,
    derive_gamma_from_nu,
    derive_nu_from_gamma,
)


# ---- SLOE Tests ----
@pytest.mark.functional
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


@pytest.mark.functional
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


@pytest.mark.functional
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
@pytest.mark.functional
def test_derive_nu_from_gamma_expected():
    mu, gamma, kappa, sigma = 0.5, 1, 0.3, 1
    expected_nu = float(np.sqrt(55) / 10)

    nu = derive_nu_from_gamma(kappa, gamma, mu, sigma)

    np.testing.assert_almost_equal(expected_nu, nu)


@pytest.mark.functional
def test_derive_gamma_from_nu_expected():
    kappa, nu, sigma, mu = 0.3, 1, 1, 0.5
    expected_gamma = float(np.sqrt(70) / 5)

    gamma = derive_gamma_from_nu(kappa, nu, sigma, mu)
    np.testing.assert_almost_equal(expected_gamma, gamma)


# --- Test log-likelihood function ---
@pytest.mark.brglm2
def test_likelihood_expected():
    """Result generated using brglm2's dbinom2 function."""
    y_adj = np.asarray([0.5, 0.5, 0.2])
    freq_weights = np.asarray([1.0, 1.0, 2.0])
    fitted_probs = np.asarray([0.8, 0.8, 0.5])

    expected_pmfs = np.asarray([0.5092958179, 0.5092958179, 0.3941805878])
    expected_log_lik = np.sum(np.log(expected_pmfs))
    expected_raw_lik = np.prod(expected_pmfs)

    actual_log_lik = compute_likelihood(y_adj, freq_weights, fitted_probs, log=True)
    actual_raw_lik = compute_likelihood(y_adj, freq_weights, fitted_probs, log=False)

    np.testing.assert_almost_equal(expected_log_lik, actual_log_lik, decimal=7)
    np.testing.assert_almost_equal(actual_raw_lik, expected_raw_lik, decimal=7)


@pytest.mark.functional
@pytest.mark.parametrize(
    "y,fw,mu",
    [
        (1.0, 1.0, 1.0),  # success == size, mu == 1 (clipped)
        (0.0, 1.0, 0.0),  # success == 0, mu == 0 (clipped)
        (
            1.0,
            1.0,
            0.0,
        ),
        (0.0, 1.0, 1.0),  # success == 0, mu clipped near 1 -> same
    ],
)
def test_compute_likelihood_finite_at_boundaries(y, fw, mu):
    y_arr = np.asarray([y])
    fw_arr = np.asarray([fw])
    mu_arr = np.asarray([mu])
    result = compute_likelihood(y_arr, fw_arr, mu_arr, log=True)
    assert np.isfinite(result)


@pytest.mark.functional
def test_compute_likelihood_returns_neg_inf_when_success_exceeds_size():
    """y > 1 (or size_i < success_i more generally) triggers the explicit
    `-np.inf` guard in compute_likelihood -- this is the only way to reach
    that branch, since success_i = y * size_i <= size_i whenever y <= 1.
    Confirms the guard fires as intended rather than being silently dead
    code, and that it degrades gracefully (returns -inf, doesn't raise)."""
    y_arr = np.asarray([1.5])  # invalid: y > 1, so success_i > size_i
    fw_arr = np.asarray([1.0])
    mu_arr = np.asarray([0.5])

    result = compute_likelihood(y_arr, fw_arr, mu_arr, log=True)

    assert result == -np.inf


@pytest.mark.functional
def test_compute_likelihood_raw_is_zero_when_success_exceeds_size():
    """Same boundary case in raw (non-log) form: exp(-inf) = 0."""
    y_arr = np.asarray([1.5])
    fw_arr = np.asarray([1.0])
    mu_arr = np.asarray([0.5])

    result = compute_likelihood(y_arr, fw_arr, mu_arr, log=False)

    assert result == 0.0


## --- Test compute taus ----
@pytest.mark.brglm2
def test_compute_taus_with_intercept():
    """Tested against brglm2 taus() function
    Intercept column will be dropped in both functions.
    """
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


@pytest.mark.brglm2
def test_compute_taus_with_no_intercept():
    """Tested against brlgm2's taus() function."""
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


# ---- Test tau compuation by computing naively ----

X_DATA = np.array(
    [
        [1, 0.5, -1.2],
        [1, -0.3, 0.8],
        [1, 1.1, 0.2],
        [1, -0.7, -0.5],
        [1, 0.2, 1.4],
        [1, 1.5, -0.9],
        [1, -1.1, 0.3],
        [1, 0.8, 0.6],
        [1, -0.4, -1.3],
        [1, 0.6, 0.1],
    ]
)


@pytest.mark.functional
def test_compute_taus_matches_manual_regression():
    """Reconstructs tau_j from first principles: regress each non-intercept
    covariate on all others via OLS, and check compute_taus reproduces the
    resulting conditional-standard-deviation value."""
    x = X_DATA
    intercept_idx = 0
    mat_x = np.delete(x, intercept_idx, axis=1)  # shape (10, 2) — slopes only
    n, p = mat_x.shape  # p = 2, matches compute_taus's internal p

    manual_taus = []
    for j in range(p):  # 0, 1 — indexes mat_x directly, no intercept to skip anymore
        other_cols = [c for c in range(p) if c != j]
        x_others = mat_x[:, other_cols]
        x_j = mat_x[:, j]
        # Find OLS coeffcients
        coef, _, _, _ = np.linalg.lstsq(x_others, x_j, rcond=None)
        # Get residuals squared
        resid = x_j - x_others @ coef
        rss = np.sum(resid**2)
        # tau_j = sqrt(RSS/ n- (p + 1))
        manual_taus.append(np.sqrt(rss / (n - p + 1)))

    manual_taus = np.asarray(manual_taus)
    actual_taus = compute_taus(x, intercept_idx)

    assert_allclose(actual_taus, manual_taus, rtol=1e-6)
