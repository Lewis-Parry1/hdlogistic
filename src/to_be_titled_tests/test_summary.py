"""tests/test_summary.py"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from to_be_titled.mdypl_fit import MDYPLModel
from to_be_titled.summary import MDYPLSummary

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


@pytest.fixture(scope="module")
def fitted_result():
    model = MDYPLModel(y=Y_DATA, x=X_DATA)
    return model.fit()


@pytest.fixture
def summ_no_hd(fitted_result):
    return MDYPLSummary(fitted_result, hd_correction=False)


@pytest.fixture
def summ_hd(fitted_result):
    return MDYPLSummary(fitted_result, hd_correction=True)


# ---------------------------------------------------------------------------
# Component-level unit tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_no_hd_correction_leaves_params_unchanged(self, fitted_result, summ_no_hd):
        assert_allclose(summ_no_hd.params, fitted_result.params)
        assert_allclose(summ_no_hd.stand_errors, fitted_result.bse)

    def test_hd_correction_changes_params(self, fitted_result, summ_hd):
        assert not np.allclose(summ_hd.params, fitted_result.params)

    def test_params_are_copies_not_views(self, fitted_result, summ_no_hd):
        summ_no_hd.params[0] = 999.0
        assert fitted_result.params[0] != 999.0


class TestHdCorrectionShapesAndNaN:
    def test_intercept_row_is_nan(self, fitted_result, summ_hd):
        idx = fitted_result.intercept_idx
        assert idx is not None
        assert np.isnan(summ_hd.stand_errors[idx])
        assert np.isnan(summ_hd.tvalues[idx])
        assert np.isnan(summ_hd.pvalues[idx])

    def test_non_intercept_rows_are_finite(self, fitted_result, summ_hd):
        idx = fitted_result.intercept_idx
        no_int = np.ones(len(summ_hd.params), dtype=bool)
        no_int[idx] = False
        assert np.all(np.isfinite(summ_hd.params[no_int]))
        assert np.all(np.isfinite(summ_hd.stand_errors[no_int]))
        assert np.all(np.isfinite(summ_hd.tvalues[no_int]))
        assert np.all(np.isfinite(summ_hd.pvalues[no_int]))

    def test_pvalues_in_valid_range(self, summ_hd, fitted_result):
        idx = fitted_result.intercept_idx
        no_int = np.ones(len(summ_hd.params), dtype=bool)
        no_int[idx] = False
        assert np.all(summ_hd.pvalues[no_int] >= 0.0)
        assert np.all(summ_hd.pvalues[no_int] <= 1.0)

    def test_tvalues_consistent_with_params_and_se(self, summ_hd, fitted_result):
        idx = fitted_result.intercept_idx
        no_int = np.ones(len(summ_hd.params), dtype=bool)
        no_int[idx] = False
        expected_t = summ_hd.params[no_int] / summ_hd.stand_errors[no_int]
        assert_allclose(summ_hd.tvalues[no_int], expected_t)

    def test_kappa_matches_p_over_nobs(self, summ_hd, fitted_result):
        p = len(summ_hd.params) - int(fitted_result.has_intercept)
        expected_kappa = p / summ_hd.nobs_eff
        assert_allclose(summ_hd.kappa, expected_kappa)

    def test_se_params_has_four_elements_with_intercept(self, summ_hd, fitted_result):
        # mu, b, sigma, intercept_estimate -- 4 elements when has_intercept=True
        assert fitted_result.has_intercept is True
        assert len(summ_hd.se_params) == 4

    def test_se_params_intercept_estimate_matches_rescaled_intercept(
        self, summ_hd, fitted_result
    ):
        idx = fitted_result.intercept_idx
        assert_allclose(
            summ_hd.se_params[3],
            summ_hd.params[idx],
            rtol=1e-10,
            err_msg=(
                "se_params intercept_estimate should equal the rescaled "
                "intercept in summ.params"
            ),
        )


class TestCoefTable:
    def test_coef_table_row_count_matches_params(self, summ_hd):
        param_names = [f"x{i}" for i in range(len(summ_hd.params))]
        table = summ_hd._build_coef_table(param_names)
        assert len(table.data) - 1 == len(summ_hd.params)

    def test_coef_table_nan_row_renders_as_nan_strings(self, summ_hd, fitted_result):
        idx = fitted_result.intercept_idx
        param_names = [f"x{i}" for i in range(len(summ_hd.params))]
        table = summ_hd._build_coef_table(param_names)
        intercept_row = table.data[idx + 1]  # +1 to skip header row
        assert intercept_row[2] == "nan"  # std err column


class TestStrRepr:
    def test_str_does_not_raise_no_hd(self, summ_no_hd):
        text = str(summ_no_hd)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_str_does_not_raise_hd(self, summ_hd):
        text = str(summ_hd)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_str_hd_contains_footer_fields(self, summ_hd):
        text = str(summ_hd)
        assert "High Dimensionality Correction applied" in text
        assert "kappa" in text

    def test_repr_matches_str(self, summ_hd):
        assert repr(summ_hd) == str(summ_hd)

    def test_construction_does_not_print(self, fitted_result, capsys):
        MDYPLSummary(fitted_result, hd_correction=True)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Integration test against R reference
# ---------------------------------------------------------------------------


def test_hd_correction_matches_r_reference(fitted_result):
    """Cross-checks HD-corrected params/se_params/kappa/signal_strength/tvalues/
    pvalues/deviance/deviance_resid/aic/nu_sloe against R's
    summary(fit, hd_correction = TRUE) on the same dataset.

    Reference values generated by the companion R script.
    """
    summ = MDYPLSummary(fitted_result, hd_correction=True)

    expected_params = np.asarray(
        [-0.803798876398766, 3.315871318793433, 0.423793541758969],
        dtype=np.float64,
    )

    expected_stand_errors = np.asarray(
        [np.nan, 1.18944105987119, 1.14029327646978], dtype=np.float64
    )

    expected_tvalues = np.asarray(
        [np.nan, 2.787755888595702, 0.371653109339544], dtype=np.float64
    )

    expected_pvalues = np.asarray(
        [np.nan, 0.00530745201030741, 0.71015114136352198], dtype=np.float64
    )

    # mu, b, sigma, intercept_estimate
    expected_se_params = np.asarray(
        [
            0.701913500972281,
            1.794149062275304,
            2.269248438727002,
            -0.803798876398766,
        ],
        dtype=np.float64,
    )

    expected_deviance_residuals = np.asarray(
        [
            1.034611,
            -0.6464056,
            0.3228017,
            -0.2643218,
            0.9921732,
            0.2115429,
            -0.1622237,
            0.4799452,
            -0.3639937,
            -1.72334,
        ],
        dtype=np.float64,
    )

    expected_kappa = 0.2
    expected_signal_strength = 6.11663335935136
    expected_deviance = 6.05054095682459
    expected_nu_sloe = 2.01083472588243
    expected_aic = 13.9046146173724

    intercept_idx = fitted_result.intercept_idx
    no_int = np.ones(len(summ.params), dtype=bool)
    no_int[intercept_idx] = False

    # --- params ---
    assert_allclose(summ.params, expected_params, rtol=1e-4)

    # --- standard errors (intercept row is NaN on both sides) ---
    assert np.isnan(summ.stand_errors[intercept_idx])
    assert_allclose(summ.stand_errors[no_int], expected_stand_errors[no_int], rtol=1e-4)

    # --- t values (intercept row is NaN on both sides) ---
    assert np.isnan(summ.tvalues[intercept_idx])
    assert_allclose(summ.tvalues[no_int], expected_tvalues[no_int], rtol=1e-4)

    # --- p values (intercept row is NaN on both sides) ---
    assert np.isnan(summ.pvalues[intercept_idx])
    assert_allclose(summ.pvalues[no_int], expected_pvalues[no_int], rtol=1e-4)

    # --- state evolution parameters (mu, b, sigma, intercept_estimate) ---
    assert len(summ.se_params) == 4, (
        f"Expected 4 state evolution parameters (mu, b, sigma, intercept_estimate), "
        f"got {len(summ.se_params)}: {summ.se_params}"
    )
    assert_allclose(
        summ.se_params[0], expected_se_params[0], rtol=1e-4, err_msg="mu mismatch"
    )
    assert_allclose(
        summ.se_params[1], expected_se_params[1], rtol=1e-4, err_msg="b mismatch"
    )
    assert_allclose(
        summ.se_params[2], expected_se_params[2], rtol=1e-4, err_msg="sigma mismatch"
    )
    assert_allclose(
        summ.se_params[3],
        expected_se_params[3],
        rtol=1e-4,
        err_msg="intercept_estimate mismatch",
    )

    # sanity: se_params intercept_estimate should equal the rescaled
    # intercept that landed in summ.params (internal consistency, not vs R)
    assert_allclose(
        summ.se_params[3],
        summ.params[intercept_idx],
        rtol=1e-10,
        err_msg="se_params intercept_estimate should equal summ.params intercept",
    )

    # --- scalar summary quantities ---
    assert_allclose(summ.kappa, expected_kappa, rtol=1e-6)
    assert_allclose(summ.signal_strength, expected_signal_strength, rtol=1e-3)
    assert_allclose(summ.deviance, expected_deviance, rtol=1e-4)
    assert_allclose(summ.nu_sloe, expected_nu_sloe, rtol=1e-4)
    assert_allclose(summ.aic, expected_aic, rtol=1e-4)

    # --- deviance residuals (per-observation vector) ---
    assert_allclose(summ.deviance_resid, expected_deviance_residuals, rtol=1e-4)
