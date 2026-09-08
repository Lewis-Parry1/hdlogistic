import numpy as np
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy.special import expit

from to_be_titled.estimation import fit_mdypl, prepare_mdypl_data
from to_be_titled.inference import compute_deviance


class TestMDYPLEstimation:
    """Tests for `prepare_mdypl_data` and `fit_mdypl` in
    `to_be_titled.estimation` with simple dataset."""

    @pytest.fixture
    def simple_data(self):
        """Small, dataset for a stable fit."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x = np.column_stack([np.ones(n), x1, x2])
        beta_true = np.array([0.2, 1.0, -0.5])
        probs = expit(x @ beta_true)
        y = rng.binomial(1, probs).astype(np.float64)
        return y, x

    def test_default_alpha_formula(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        n, p = x.shape

        result = fit_mdypl(data, alpha=None)

        expected_alpha = data.nobs_eff / (
            data.nobs_eff + data.rank - int(data.has_intercept)
        )
        assert_allclose(result.alpha, expected_alpha)

    @pytest.mark.functional
    def test_fixed_alpha_is_respected(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)

        result = fit_mdypl(data, alpha=0.7)
        assert result.alpha == 0.7

    @pytest.mark.functional
    def test_alpha_out_of_range_raises(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)

        with pytest.raises(ValueError, match="Shrinkage parameter"):
            fit_mdypl(data, alpha=1.5)

    @pytest.mark.functional
    def test_rank_deficient_design_raises(self, simple_data):
        y, x = simple_data
        x_deficient = np.column_stack(
            [x, x[:, 1]]
        )  # duplicate column -> rank-deficient

        with pytest.raises(ValueError, match="rank-deficient"):
            prepare_mdypl_data(x=x_deficient, y=y)

    @pytest.mark.functional
    def test_intercept_detected_correctly(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)

        assert data.has_intercept is True
        assert data.intercept_idx == 0

    @pytest.mark.statsmodels
    def test_alpha_equals_one_matches_standard_logistic_regression(self, simple_data):
        """With alpha=1, y_adj == y exactly, so MDYPL should reduce to an
        ordinary logistic regression fit."""
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)

        mdypl_result = fit_mdypl(data, alpha=1.0)
        standard_result = sm.GLM(y, x, family=sm.families.Binomial()).fit()

        assert_allclose(
            mdypl_result.params,
            standard_result.params,
            rtol=1e-6,
            err_msg="MDYPL with alpha=1 should exactly match standard "
            "logistic regression",
        )

    @pytest.mark.statsmodels
    def test_general_alpha_matches_manual_y_adj_fit(self, simple_data):
        """Reconstructs the MDYPL fit by hand: manually shrinks the response
        toward 0.5 using a fixed alpha, fits an ordinary GLM on the adjusted
        response, and checks fit_mdypl reproduces it exactly. 
        
        This directly tests the definitional equivalence between MDYPL fitting 
        and an ordinary GLM fit on y_adj, at a non-degenerate alpha (unlike the
        alpha=1 tests above, which don't exercise the response adjustment
        logic at all)."""
        y, x = simple_data
        alpha = 0.3
        data = prepare_mdypl_data(x=x, y=y)

        mdypl_result = fit_mdypl(data, alpha=alpha)

        y_adj_manual = alpha * y + (1.0 - alpha) / 2.0
        manual_fit = sm.GLM(y_adj_manual, x, family=sm.families.Binomial()).fit()

        assert_allclose(mdypl_result.params, manual_fit.params, rtol=1e-6)
        assert_allclose(mdypl_result.deviance_adj, manual_fit.deviance, rtol=1e-6)

    @pytest.mark.functional
    def test_cov_params_matches_manual_fisher_information(self, simple_data):
        """Reconstructs the covariance matrix from first principles (weighted
        normal equations) rather than trusting statsmodels' internal cov_params,
        mirroring brglm2's SE test."""
        y, x = simple_data
        a = 0.3
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=a)

        mu = result.fitted_probs
        working_weights = data.weights * mu * (1.0 - mu)
        fisher_info = x.T @ (x * working_weights[:, None])
        manual_cov = np.linalg.inv(fisher_info)

        assert_allclose(np.diag(manual_cov), np.diag(result.cov_params), rtol=1e-5)

    # --- Weight and Offset Length Checks ---
    @pytest.mark.functional
    def test_weights_wrong_length_raises(self, simple_data):
        y, x = simple_data
        with pytest.raises(ValueError, match="weights length"):
            prepare_mdypl_data(x=x, y=y, weights=np.ones(len(y) - 1))

    @pytest.mark.functional
    def test_offset_wrong_length_raises(self, simple_data):
        y, x = simple_data
        with pytest.raises(ValueError, match="offset length"):
            prepare_mdypl_data(x=x, y=y, offset=np.ones(len(y) - 1))

    @pytest.mark.functional
    def test_scalar_offset_is_broadcast(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y, offset=np.array([0.5]))
        assert data.offset is not None
        assert data.offset.shape == (len(y),)
        assert_allclose(data.offset, np.full(len(y), 0.5))

    @pytest.mark.functional
    def test_offset_accepts_bare_float(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y, offset=0.5)

        assert data.offset is not None
        assert data.offset.shape == (len(y),)
        assert_allclose(data.offset, np.full(len(y), 0.5))

    @pytest.mark.functional
    def test_no_weights_defaults_to_ones(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        assert_allclose(data.weights, np.ones(len(y)))

    @pytest.mark.functional
    def test_no_offset_defaults_to_none(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        assert data.offset is None

    @pytest.mark.functional
    def test_nobs_eff_equals_sum_of_weights(self, simple_data):
        y, x = simple_data
        rng = np.random.default_rng(2)
        weights = rng.uniform(0.5, 2.0, size=len(y))
        data = prepare_mdypl_data(x=x, y=y, weights=weights)
        assert_allclose(data.nobs_eff, np.sum(weights))

    @pytest.mark.functional
    def test_rank_equals_number_of_columns(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        assert data.rank == x.shape[1]

    @pytest.mark.statsmodels
    def test_deviance_matches_statsmodels_at_alpha_one(self, simple_data):
        """At alpha=1, y_adj == y, so deviance should match statsmodels' own
        deviance computation exactly."""
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=1.0)
        deviance = result.deviance_adj

        standard_result = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        assert_allclose(deviance, standard_result.deviance, rtol=1e-5)

    @pytest.mark.functional
    def test_resid_deviance_shape(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=0.8)
        assert result.resid_deviance_adj.shape == (len(y),)

    @pytest.mark.functional
    def test_deviance_residuals_sum_of_squares_equals_deviance(self, simple_data):
        """Sanity check on the mathematical relationship: sum(resid_deviance**2)
        should equal `deviance`."""
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=0.8)
        assert_allclose(
            np.sum(result.resid_deviance_adj**2), result.deviance_adj, rtol=1e-6
        )

    @pytest.fixture
    def no_intercept_data(self):
        rng = np.random.default_rng(7)
        n = 100
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x = np.column_stack([x1, x2])  # no intercept column
        beta_true = np.array([1.0, -0.5])
        probs = expit(x @ beta_true)
        y = rng.binomial(1, probs).astype(np.float64)
        return y, x

    @pytest.mark.functional
    def test_no_intercept_detected_correctly(self, no_intercept_data):
        y, x = no_intercept_data
        data = prepare_mdypl_data(x=x, y=y)
        assert data.has_intercept is False
        assert data.intercept_idx is None

    def test_no_intercept_null_fitted_probs_is_half_without_offset(
        self, no_intercept_data
    ):
        y, x = no_intercept_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=1.0)

        assert_allclose(result.null_fitted_probs, np.full(len(y), 0.5))

        expected_null_deviance = compute_deviance(
            data.y_raw, np.full(len(y), 0.5), data.weights, eps=1e-15
        )
        assert_allclose(result.null_deviance_adj, expected_null_deviance)

    @pytest.mark.functional
    def test_no_intercept_property_is_none(self, no_intercept_data):
        y, x = no_intercept_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=1.0)
        assert result.intercept is None

    @pytest.mark.functional
    def test_aic_consistent_with_llf(self, simple_data):
        """Guards the llf/aic decoupling: aic must always equal
        -2 * llf + 2 * rank, regardless of how each was computed."""
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=0.8)
        assert_allclose(result.aic, -2.0 * result.llf + 2.0 * result.rank, rtol=1e-10)

    @pytest.mark.functional
    def test_sloe_returns_finite_scalar(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=0.8)
        assert isinstance(result.sloe, float)
        assert np.isfinite(result.sloe)

    @pytest.mark.statsmodels
    def test_common_values_match_statsmodels_at_alpha_one(self, simple_data):
        """At alpha=1, y_adj == y exactly, so MDYPL should reduce to an ordinary
        logistic regression fit. Every commonly-used fitted quantity should
        match statsmodels' own GLM computation."""
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=1.0)

        standard_result = sm.GLM(y, x, family=sm.families.Binomial()).fit()

        assert_allclose(
            result.params,
            standard_result.params,
            rtol=1e-6,
            err_msg="params mismatch at alpha=1",
        )
        assert_allclose(
            result.fitted_probs,
            standard_result.fittedvalues,
            rtol=1e-6,
            err_msg="fitted_probs mismatch at alpha=1",
        )
        assert_allclose(
            result.linear_predictors,
            standard_result.predict(which="linear"),
            rtol=1e-6,
            err_msg="linear_predictors mismatch at alpha=1",
        )
        assert_allclose(
            result.deviance_adj,
            standard_result.deviance,
            rtol=1e-5,
            err_msg="deviance mismatch at alpha=1",
        )
        assert_allclose(
            result.null_deviance_adj,
            standard_result.null_deviance,
            rtol=1e-5,
            err_msg="null_deviance mismatch at alpha=1",
        )
        assert_allclose(
            result.aic,
            standard_result.aic,
            rtol=1e-5,
            err_msg="aic mismatch at alpha=1",
        )
        assert_allclose(
            result.resid_deviance_adj,
            standard_result.resid_deviance,
            rtol=1e-5,
            err_msg="resid_deviance mismatch at alpha=1",
        )
        assert_allclose(
            result.llf,
            standard_result.llf,
            rtol=1e-5,
            err_msg="llf mismatch at alpha=1",
        )

        assert result.converged == standard_result.converged

    @pytest.fixture
    def high_dim_data(self):
        """p/n ratio and signal strength calibrated to approach the
        high-dimensional regime the MDYPL estimator is designed for.

        beta is normalised so that var(x_i^T beta) = beta^T Var(X) beta = gamma**2
        exactly (population-level), since X has i.i.d. unit-variance columns and
        beta^T beta = gamma**2 when ||beta|| = gamma.
        """
        rng = np.random.default_rng(11)
        n, p_slopes = 60, 39
        x_slopes = rng.normal(size=(n, p_slopes))
        x = np.column_stack([np.ones(n), x_slopes])

        gamma = 2
        beta_raw = rng.normal(size=p_slopes)
        beta_slopes = beta_raw * (gamma / np.linalg.norm(beta_raw))

        intercept_true = 0.0
        with np.errstate(all="ignore"):
            linear_predictor = intercept_true + x_slopes @ beta_slopes
        probs = expit(linear_predictor)

        y = rng.binomial(1, probs).astype(np.float64)
        return y, x

    @pytest.mark.functional
    def test_converges_in_high_dimensional_regime(self, high_dim_data):
        """The core motivating case for MDYPL: fit_mdypl should still produce
        finite, converged estimates where ordinary MLE would be expected to
        struggle (large p/n, strong signal)."""
        y, x = high_dim_data
        data = prepare_mdypl_data(x=x, y=y)

        with np.errstate(all="ignore"):
            result = fit_mdypl(data, alpha=None)

        assert result.converged
        assert result.params.shape == (x.shape[1],)
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.aic)
        assert np.isfinite(result.llf)
        assert np.isfinite(result.deviance_adj)


## --- Result cross check with brglm2 ---
X_DATA = np.array(
    [
        [1.0, 0.5, -1.2],
        [1.0, -0.3, 0.8],
        [1.0, 1.1, 0.2],
        [1.0, -0.7, -0.5],
        [1.0, 0.2, 1.4],
        [1.0, 1.5, -0.9],
        [1.0, -1.1, 0.3],
        [1.0, 0.8, 0.6],
        [1.0, -0.4, -1.3],
        [1.0, 0.6, 0.1],
    ]
)
Y_DATA = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.float64)

@pytest.mark.brglm2
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {},
            dict(
                alpha=0.8333333,
                params=np.asarray([-0.5691641, 2.3274548, 0.2974664]),
                deviance=3.268387,
                null_deviance=8.126224,
                aic=13.49387,
                leverages=np.asarray([0.4939179, 0.3314687, 0.2039765, 0.2356371,
                                      0.5064856, 0.2523204, 0.1937260, 0.2242109,
                                      0.3728649, 0.1853921]),
                resid=np.asarray([0.79290616, -0.45690657, 0.10078244, -0.01413417,
                                  0.75916487, -0.06840387, 0.16265343, 0.29149666,
                                  -0.15171620, -1.30579923]),
            ),
        ),
        (
            dict(
                weights=np.array([1.2, 0.8, 1.5, 0.6, 1.0, 1.3, 0.9, 1.1, 0.7, 1.4]),
                offset=np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.05, 0.2, -0.15, 0.1]),
            ),
            dict(
                alpha=0.84,
                params=np.asarray([-0.5855203, 2.1187811, 0.2748631]),
                deviance=4.152915,
                null_deviance=8.204773,
                aic=14.38138,
                leverages=np.asarray([0.5365223, 0.2632042, 0.2706447, 0.2262324,
                                      0.5129572, 0.2901187, 0.2168195, 0.2004426,
                                      0.2776447, 0.2054137]),
                resid=np.asarray([0.87540945, -0.35848960, 0.21396128, -0.12155933,
                                  0.85392097, -0.01625093, 0.10688309, 0.31573416,
                                  -0.12231564, -1.53035414]),
            ),
        ),
    ],
    ids=["no_weights_no_offset", "with_weights_and_offset"],
)
def test_fitted_params_match_r_reference(kwargs, expected):
    """Cross-check against R's mdyplFit() on a fixed, hand-written dataset."""
    data = prepare_mdypl_data(x=X_DATA, y=Y_DATA, **kwargs)
    result = fit_mdypl(data, alpha=None)

    assert_allclose(result.alpha, expected["alpha"], rtol=1e-6)
    assert_allclose(result.params, expected["params"], rtol=1e-6)
    assert_allclose(result.deviance_adj, expected["deviance"], rtol=1e-6)
    assert_allclose(result.aic, expected["aic"], rtol=1e-6)
    # Slight discrepency in the way R fits the null model compared to this package
    # In order to get the same result, alpha must be fixed to default value
    # in R. If it defaults then this causes null to be fit with alpha =1. 
    assert_allclose(result.null_deviance_adj, expected["null_deviance"], rtol=1e-6)
    assert_allclose(result.leverages, expected["leverages"], rtol=1e-6)
    assert_allclose(result.resid_deviance_adj, expected["resid"], rtol=1e-6)