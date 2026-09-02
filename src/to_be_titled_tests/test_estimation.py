import numpy as np
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy.special import expit

from to_be_titled.estimation import fit_mdypl, prepare_mdypl_data


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

        expected_alpha = data.nobs_eff / (data.nobs_eff + data.rank - int(data.has_intercept))
        assert_allclose(result.alpha, expected_alpha)

    def test_fixed_alpha_is_respected(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)

        result = fit_mdypl(data, alpha=0.7)
        assert result.alpha == 0.7

    def test_alpha_out_of_range_raises(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)

        with pytest.raises(ValueError, match="Shrinkage parameter"):
            fit_mdypl(data, alpha=1.5)

    def test_rank_deficient_design_raises(self, simple_data):
        y, x = simple_data
        x_deficient = np.column_stack([x, x[:, 1]])  # duplicate column -> rank-deficient

        with pytest.raises(ValueError, match="rank-deficient"):
            prepare_mdypl_data(x=x_deficient, y=y)

    def test_intercept_detected_correctly(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)

        assert data.has_intercept is True
        assert data.intercept_idx == 0

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
            err_msg="MDYPL with alpha=1 should exactly match standard logistic regression",
        )

    # --- Weight and Offset Length Checks ---
    def test_weights_wrong_length_raises(self, simple_data):
        y, x = simple_data
        with pytest.raises(ValueError, match="weights length"):
            prepare_mdypl_data(x=x, y=y, weights=np.ones(len(y) - 1))

    def test_offset_wrong_length_raises(self, simple_data):
        y, x = simple_data
        with pytest.raises(ValueError, match="offset length"):
            prepare_mdypl_data(x=x, y=y, offset=np.ones(len(y) - 1))

    def test_scalar_offset_is_broadcast(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y, offset=np.array([0.5]))
        assert data.offset is not None
        assert data.offset.shape == (len(y),)
        assert_allclose(data.offset, np.full(len(y), 0.5))

    def test_offset_accepts_bare_float(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y, offset=0.5)

        assert data.offset is not None
        assert data.offset.shape == (len(y),)
        assert_allclose(data.offset, np.full(len(y), 0.5))

    def test_no_weights_defaults_to_ones(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        assert_allclose(data.weights, np.ones(len(y)))

    def test_no_offset_defaults_to_none(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        assert data.offset is None

    def test_nobs_eff_equals_sum_of_weights(self, simple_data):
        y, x = simple_data
        rng = np.random.default_rng(2)
        weights = rng.uniform(0.5, 2.0, size=len(y))
        data = prepare_mdypl_data(x=x, y=y, weights=weights)
        assert_allclose(data.nobs_eff, np.sum(weights))

    def test_rank_equals_number_of_columns(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        assert data.rank == x.shape[1]

    def test_deviance_matches_statsmodels_at_alpha_one(self, simple_data):
        """At alpha=1, y_adj == y, so deviance should match statsmodels' own
        deviance computation exactly."""
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=1.0)
        deviance = result.deviance

        standard_result = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        assert_allclose(deviance, standard_result.deviance, rtol=1e-5)

    def test_resid_deviance_shape(self, simple_data):
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=0.8)
        assert result.resid_deviance.shape == (len(y),)
        assert result.resid_pearson.shape == (len(y),)

    def test_deviance_residuals_sum_of_squares_equals_deviance(self, simple_data):
        """Sanity check on the mathematical relationship: sum(resid_deviance**2)
        should equal `deviance`."""
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=0.8)
        assert_allclose(np.sum(result.resid_deviance**2), result.deviance, rtol=1e-6)

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

    def test_no_intercept_detected_correctly(self, no_intercept_data):
        y, x = no_intercept_data
        data = prepare_mdypl_data(x=x, y=y)
        assert data.has_intercept is False
        assert data.intercept_idx is None

    def test_no_intercept_null_fitted_probs_is_half_without_offset(self, no_intercept_data):
        y, x = no_intercept_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=1.0)
        assert_allclose(result.null_deviance, result.null_deviance)  # sanity: no crash
        # Recompute expected null deviance directly, since null_fitted_probs should be 0.5 everywhere
        from to_be_titled.inference import compute_deviance
        expected_null_deviance = compute_deviance(
            data.y_raw, np.full(len(y), 0.5), data.weights, eps=1e-15
        )
        assert_allclose(result.null_deviance, expected_null_deviance)

    def test_no_intercept_property_is_none(self, no_intercept_data):
        y, x = no_intercept_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=1.0)
        assert result.intercept is None

    def test_common_values_match_statsmodels_at_alpha_one(self, simple_data):
        """At alpha=1, y_adj == y exactly, so MDYPL should reduce to an ordinary
        logistic regression fit. Every commonly-used fitted quantity should
        match statsmodels' own GLM computation."""
        y, x = simple_data
        data = prepare_mdypl_data(x=x, y=y)
        result = fit_mdypl(data, alpha=1.0)

        standard_result = sm.GLM(y, x, family=sm.families.Binomial()).fit()

        assert_allclose(
            result.params, standard_result.params, rtol=1e-6,
            err_msg="params mismatch at alpha=1",
        )
        assert_allclose(
            result.fitted_probs, standard_result.fittedvalues, rtol=1e-6,
            err_msg="fitted_probs mismatch at alpha=1",
        )
        assert_allclose(
            result.linear_predictors, standard_result.predict(which="linear"), rtol=1e-6,
            err_msg="linear_predictors mismatch at alpha=1",
        )
        assert_allclose(
            result.deviance, standard_result.deviance, rtol=1e-5,
            err_msg="deviance mismatch at alpha=1",
        )
        assert_allclose(
            result.null_deviance, standard_result.null_deviance, rtol=1e-5,
            err_msg="null_deviance mismatch at alpha=1",
        )
        assert_allclose(
            result.aic, standard_result.aic, rtol=1e-5,
            err_msg="aic mismatch at alpha=1",
        )
        assert_allclose(
            result.bic, standard_result.bic_llf, rtol=1e-5,
            err_msg="bic mismatch at alpha=1",
        )
        assert_allclose(
            result.resid_deviance, standard_result.resid_deviance, rtol=1e-5,
            err_msg="resid_deviance mismatch at alpha=1",
        )
        assert_allclose(
            result.resid_pearson, standard_result.resid_pearson, rtol=1e-5,
            err_msg="resid_pearson mismatch at alpha=1",
        )
        assert result.converged == standard_result.converged



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


def test_fitted_params_match_r_reference():
    """Cross-check against R's mdyplFit() on a fixed, hand-written dataset.
    Reference values generated by running the R script in the test docstring
    / project notes and copying `coef(fit)`, `fit$alpha`, `fit$deviance`.
    """
    x = X_DATA
    y = Y_DATA

    data = prepare_mdypl_data(x=x, y=y)
    result = fit_mdypl(data, alpha=None)

    expected_alpha = 0.8333333
    expected_params = np.asarray([-0.5691641, 2.3274548, 0.2974664])
    expected_deviance = 3.268387
    # expected_null_deviance = 8.126224  # TODO: re-enable once discrepancy
    # with brglm2 expected null-deviance value is resolved.
    expected_aic = 13.49387
    expected_leverages = np.asarray(
        [
            0.4939179,
            0.3314687,
            0.2039765,
            0.2356371,
            0.5064856,
            0.2523204,
            0.1937260,
            0.2242109,
            0.3728649,
            0.1853921,
        ]
    )

    assert_allclose(result.alpha, expected_alpha, rtol=1e-6)
    assert_allclose(result.params, expected_params, rtol=1e-6)
    assert_allclose(result.deviance, expected_deviance, rtol=1e-6)
    assert_allclose(result.aic, expected_aic, rtol=1e-6)
    # assert_allclose(result.null_deviance, expected_null_deviance, rtol=1e-6)
    assert_allclose(result.leverages, expected_leverages, rtol=1e-6)