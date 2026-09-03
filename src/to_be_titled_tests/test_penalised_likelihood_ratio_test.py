import numpy as np
<<<<<<< Updated upstream
from numpy.testing import assert_allclose

from to_be_titled.estimation import fit_mdypl, prepare_mdypl_data
from to_be_titled.penalised_likelihood_ratio_test import penalised_lrt
=======
import pytest
from to_be_titled.mdypl_fit import MDYPLModel

from to_be_titled.penalised_likelihood_ratio_test import (
    PenalisedLRTResults,
    penalised_lrt,
)
from to_be_titled.types import MDYPLResults
>>>>>>> Stashed changes

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


def test_penalised_lrt_matches_r_reference_no_hd():
    """Cross-check `penalised_lrt` (hd_correction=False) against R's
    plrtest.mdyplFit() on the same fixed 10-observation dataset, comparing
    a reduced model (intercept + x1) against a full model
    (intercept + x1 + x2), both fit with a fixed alpha=0.8 so that
    plrtest's identical-alpha requirement is satisfied.
    """
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]

    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA)
    data_full = prepare_mdypl_data(x=x_full, y=Y_DATA)

    fit_reduced = fit_mdypl(data_reduced, alpha=0.8)
    fit_full = fit_mdypl(data_full, alpha=0.8)

    assert_allclose(fit_reduced.deviance, 3.062321, rtol=1e-5)
    assert_allclose(fit_full.deviance, 2.956846, rtol=1e-5)
    assert fit_reduced.rank == 2
    assert fit_full.rank == 3

    result = penalised_lrt(fit_reduced, fit_full, hd_correction=False)

    expected_statistic = 0.105474361567877
    expected_df = 1
    expected_p_value = 0.745356539561018

    assert_allclose(result.statistic, expected_statistic, rtol=1e-6)
    assert result.df == expected_df
    assert_allclose(result.p_value, expected_p_value, rtol=1e-6)


def test_penalised_lrt_matches_r_reference_with_hd():
    """Cross-check `penalised_lrt` (hd_correction=True) against R's
    plrtest.mdyplFit(..., hd_correction = TRUE), same reduced/full
    setup as the non-HD test above.
    """
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]

    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA)
    data_full = prepare_mdypl_data(x=x_full, y=Y_DATA)

    fit_reduced = fit_mdypl(data_reduced, alpha=0.8)
    fit_full = fit_mdypl(data_full, alpha=0.8)

    result = penalised_lrt(fit_reduced, fit_full, hd_correction=True)

    expected_statistic = 0.216012608750709
    expected_df = 1
    expected_p_value = 0.642095051893411
    expected_kappa = 0.2
    expected_signal_strength = 6.81404005270763
    expected_se_params = np.asarray(
        [0.616976540591563, 1.695052143520944, 2.034278622083952]
    )

    assert_allclose(result.statistic, expected_statistic, rtol=1e-6)
    assert result.df == expected_df
    assert_allclose(result.p_value, expected_p_value, rtol=1e-6)

    assert result.kappa is not None
    assert result.signal_strength is not None
    assert result.se_params is not None

    assert_allclose(result.kappa, expected_kappa, rtol=1e-6)
    assert_allclose(result.signal_strength, expected_signal_strength, rtol=1e-6)
    assert_allclose(result.se_params[:3], expected_se_params, rtol=1e-6)
