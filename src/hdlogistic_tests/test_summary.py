import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from hdlogistic.estimation import fit_mdypl, prepare_mdypl_data
from hdlogistic.inference import compute_aic, compute_likelihood, compute_sloe
from hdlogistic.solvers import solve_state_equation
from hdlogistic.summary import summary

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

Y_DATA = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.float64)


@pytest.mark.functional
def test_summary_shapes_and_types_hd_correction():
    data = prepare_mdypl_data(x=X_DATA, y=Y_DATA)
    result = fit_mdypl(data, alpha=None)
    hd_summ = summary(result, high_dimensional_correction=True)

    p = X_DATA.shape[1]
    assert hd_summ.params.shape == (p,)
    assert hd_summ.bse.shape == (p,)
    assert hd_summ.zvalues.shape == (p,)
    assert hd_summ.pvalues.shape == (p,)

    assert isinstance(hd_summ.deviance_adj, float)
    assert isinstance(hd_summ.deviance_raw, float)
    assert isinstance(hd_summ.null_deviance_adj, float)
    assert isinstance(hd_summ.null_deviance_raw, float)
    assert np.isfinite(hd_summ.deviance_adj)
    assert np.isfinite(hd_summ.deviance_raw)

    assert isinstance(hd_summ.aic, float)
    assert isinstance(hd_summ.llf, float)

    assert hd_summ.hd_diagnostics is not None
    assert isinstance(hd_summ.hd_diagnostics.kappa, float)


@pytest.mark.functional
def test_summary_hd_correction_false_has_no_diagnostics():
    data = prepare_mdypl_data(x=X_DATA, y=Y_DATA)
    result = fit_mdypl(data, alpha=None)
    summ = summary(result, high_dimensional_correction=False)
    assert summ.hd_diagnostics is None


@pytest.mark.functional
def test_summary_aic_consistent_with_llf():
    data = prepare_mdypl_data(x=X_DATA, y=Y_DATA)
    result = fit_mdypl(data, alpha=None)
    hd_summ = summary(result, high_dimensional_correction=True)
    assert_allclose(hd_summ.aic, -2.0 * hd_summ.llf + 2.0 * result.rank, rtol=1e-10)


@pytest.mark.functional
def test_summary_aic_matches_manual_reconstruction_from_rescaled_params():
    data = prepare_mdypl_data(x=X_DATA, y=Y_DATA)
    result = fit_mdypl(data, alpha=None)
    hd_summ = summary(result, high_dimensional_correction=True)

    manual_probs = expit(X_DATA @ hd_summ.params)
    manual_llf = compute_likelihood(
        result.y_adj, result.weights, manual_probs, log=True
    )
    manual_aic = compute_aic(manual_llf, result.rank)

    assert_allclose(hd_summ.aic, manual_aic, rtol=1e-8)


@pytest.mark.functional
def test_summary_se_params_matches_direct_solver_call():
    """Reconstructs the state-evolution solve independently of summary(),
    checking summary() wires up solve_state_equation with the same
    arguments/results it would get calling the solver directly"""

    data = prepare_mdypl_data(x=X_DATA, y=Y_DATA)
    result = fit_mdypl(data, alpha=None)
    hd_summ = summary(result, high_dimensional_correction=True)

    nu_sloe = compute_sloe(
        result.y_adj, result.linear_predictors, result.fitted_probs, result.leverages
    )
    p = X_DATA.shape[1] - 1
    kappa = p / result.nobs_eff

    se_params_direct, _ = solve_state_equation(
        kappa=kappa,
        signal_strength=nu_sloe,
        alpha=result.alpha,
        intercept=result.intercept,
        corrupted=True,
    )

    assert hd_summ.hd_diagnostics is not None
    assert_allclose(
        hd_summ.hd_diagnostics.se_params,
        se_params_direct.solution.to_array(),
        rtol=1e-8,
    )


@pytest.mark.brglm2
def test_hd_corrected_summary_matches_r_reference() -> None:
    """Cross-check high-dimensional-corrected summary() output against R's
    summary.mdyplFit(fit, hd_correction = TRUE) on the same fixed
    10-observation dataset used in the other reference tests.
    """

    data = prepare_mdypl_data(x=X_DATA, y=Y_DATA)
    result = fit_mdypl(data, alpha=None)
    hd_summ = summary(result, high_dimensional_correction=True)

    expected_params = np.asarray(
        [-0.803798876398766, 3.315871318793433, 0.423793541758969]
    )
    expected_bse = np.asarray([np.nan, 1.18944105987119, 1.14029327646978])
    expected_zvalues = np.asarray([np.nan, 2.787755888595702, 0.371653109339544])
    expected_pvalues = np.asarray([np.nan, 0.00530745201030741, 0.71015114136352198])
    expected_se_params = np.asarray(
        [0.701913500972281, 1.794149062275304, 2.269248438727002, -0.803798876398766]
    )
    expected_kappa = 0.2
    expected_signal_strength = 6.11663335935136
    expected_deviance = 6.05054095682459
    expected_resid_deviance = np.asarray(
        [
            1.034610841246200,
            -0.646405576548440,
            0.322801655339334,
            -0.264321828338061,
            0.992173216780434,
            0.211542872258388,
            -0.162223670490564,
            0.479945217726871,
            -0.363993684169680,
            -1.723340025845221,
        ]
    )
    expected_aic = 13.9046146173724
    expected_nu_sloe = 2.01083472588243

    assert_allclose(hd_summ.params, expected_params, rtol=1e-6)
    assert_allclose(hd_summ.bse, expected_bse, rtol=1e-6, equal_nan=True)
    assert_allclose(hd_summ.zvalues, expected_zvalues, rtol=1e-6, equal_nan=True)
    assert_allclose(hd_summ.pvalues, expected_pvalues, rtol=1e-6, equal_nan=True)

    assert hd_summ.hd_diagnostics is not None
    assert_allclose(hd_summ.hd_diagnostics.se_params, expected_se_params, rtol=1e-6)
    assert_allclose(hd_summ.hd_diagnostics.kappa, expected_kappa, rtol=1e-6)
    assert_allclose(
        hd_summ.hd_diagnostics.signal_strength, expected_signal_strength, rtol=1e-6
    )
    assert_allclose(hd_summ.hd_diagnostics.nu_sloe, expected_nu_sloe, rtol=1e-6)

    assert_allclose(hd_summ.deviance_raw, expected_deviance, rtol=1e-6)
    assert_allclose(hd_summ.resid_deviance_raw, expected_resid_deviance, rtol=1e-6)
    assert_allclose(hd_summ.aic, expected_aic, rtol=1e-6)


@pytest.mark.brglm2
def test_hd_corrected_summary_matches_r_reference_with_weights_and_offset():
    """Cross-check high-dimensional-corrected summary() output against R's
    summary.mdyplFit(fit, hd_correction = TRUE), fit with weights and an
    offset, on the same fixed 10-observation dataset used in the other
    reference tests. Both `weights_match` and `offset_match` printed TRUE
    on the R side, confirming R retained the exact inputs used here.
    """
    weights = np.array([1.2, 0.8, 1.5, 0.6, 1.0, 1.3, 0.9, 1.1, 0.7, 1.4])
    offset = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.05, 0.2, -0.15, 0.1])

    data = prepare_mdypl_data(x=X_DATA, y=Y_DATA, weights=weights, offset=offset)
    result = fit_mdypl(data, alpha=None)
    hd_summ = summary(result, high_dimensional_correction=True)

    expected_params = np.asarray(
        [-0.772004562718499, 2.819554992209595, 0.365772432745014]
    )
    expected_bse = np.asarray([np.nan, 1.09853868770901, 1.05314699634816])
    expected_zvalues = np.asarray([np.nan, 2.566641506353996, 0.347313750134928])
    expected_pvalues = np.asarray([np.nan, 0.0102688716479713, 0.7283556218810713])
    expected_se_params = np.asarray(
        [0.751459400416222, 1.622533520155142, 2.299169821844732, -0.772004562718499]
    )
    expected_kappa = 0.19047619047619
    expected_signal_strength = 4.67681318924516
    # TODO: Uncomment when offset issue fixed in MDYPL Summary
    """expected_deviance = 7.47500655661417
    expected_resid_deviance = np.asarray(
        [
            1.198657768757707,
            -0.614040422832868,
            0.509750373916438,
            -0.250021475926415,
            1.051342125150422,
            0.333849233234426,
            -0.203163947019451,
            0.606692847688290,
            -0.352811360117789,
            -1.894260028540432,
        ]
    )
    expected_aic = 14.5077551413619"""
    expected_nu_sloe = 1.90993381540636

    assert_allclose(hd_summ.params, expected_params, rtol=1e-6)
    assert_allclose(hd_summ.bse, expected_bse, rtol=1e-6, equal_nan=True)
    assert_allclose(hd_summ.zvalues, expected_zvalues, rtol=1e-6, equal_nan=True)
    assert_allclose(hd_summ.pvalues, expected_pvalues, rtol=1e-6, equal_nan=True)

    assert hd_summ.hd_diagnostics is not None
    assert_allclose(hd_summ.hd_diagnostics.se_params, expected_se_params, rtol=1e-6)
    assert_allclose(hd_summ.hd_diagnostics.kappa, expected_kappa, rtol=1e-6)
    assert_allclose(
        hd_summ.hd_diagnostics.signal_strength, expected_signal_strength, rtol=1e-6
    )
    assert_allclose(hd_summ.hd_diagnostics.nu_sloe, expected_nu_sloe, rtol=1e-6)

    # TODO: Add tests back once R bug has been resolved. Deviance, resid_deviance
    # and aic
