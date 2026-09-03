# TODO: Lewis, smaller unit tests needed (shape, type etc.)

import numpy as np
from numpy.testing import assert_allclose

from to_be_titled.estimation import fit_mdypl, prepare_mdypl_data
from to_be_titled.summary import summary
from to_be_titled.mdypl_fit import MDYPLModel

from to_be_titled.summary import MDYPLSummary

# TODO: I end up creating this same toy dataset 3 times maybe
# We should just have a dataset file in tests?
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
        [0.701913500972281, 1.794149062275304, 2.269248438727002]
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
    assert_allclose(hd_summ.hd_diagnostics.se_params[:3], expected_se_params, rtol=1e-6)
    assert_allclose(hd_summ.hd_diagnostics.kappa, expected_kappa, rtol=1e-6)
    assert_allclose(
        hd_summ.hd_diagnostics.signal_strength, expected_signal_strength, rtol=1e-6
    )
    assert_allclose(hd_summ.hd_diagnostics.nu_sloe, expected_nu_sloe, rtol=1e-6)

    assert_allclose(hd_summ.deviance, expected_deviance, rtol=1e-6)
    assert_allclose(hd_summ.resid_deviance, expected_resid_deviance, rtol=1e-6)
    assert_allclose(hd_summ.aic, expected_aic, rtol=1e-6)


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
        [0.751459400416222, 1.622533520155142, 2.299169821844732]
    )
    expected_kappa = 0.19047619047619
    expected_signal_strength = 4.67681318924516
    expected_deviance = 7.47500655661417
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
    expected_aic = 14.5077551413619
    expected_nu_sloe = 1.90993381540636

    assert_allclose(hd_summ.params, expected_params, rtol=1e-6)
    assert_allclose(hd_summ.bse, expected_bse, rtol=1e-6, equal_nan=True)
    assert_allclose(hd_summ.zvalues, expected_zvalues, rtol=1e-6, equal_nan=True)
    assert_allclose(hd_summ.pvalues, expected_pvalues, rtol=1e-6, equal_nan=True)

    assert hd_summ.hd_diagnostics is not None
    assert_allclose(hd_summ.hd_diagnostics.se_params[:3], expected_se_params, rtol=1e-6)
    assert_allclose(hd_summ.hd_diagnostics.kappa, expected_kappa, rtol=1e-6)
    assert_allclose(
        hd_summ.hd_diagnostics.signal_strength, expected_signal_strength, rtol=1e-6
    )
    assert_allclose(hd_summ.hd_diagnostics.nu_sloe, expected_nu_sloe, rtol=1e-6)
    # TODO: Add tests back once R bug has been resolved.
    # assert_allclose(hd_summ.deviance, expected_deviance, rtol=1e-6)
    # assert_allclose(hd_summ.resid_deviance, expected_resid_deviance, rtol=1e-6)
    # assert_allclose(hd_summ.aic, expected_aic, rtol=1e-6)
