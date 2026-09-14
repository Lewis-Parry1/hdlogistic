import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hdlogistic.estimation import fit_mdypl, prepare_mdypl_data
from hdlogistic.penalised_likelihood_ratio_test import penalised_lrt

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
def test_penalised_lrt_raises_on_alpha_mismatch():
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]
    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA)
    data_full = prepare_mdypl_data(x=x_full, y=Y_DATA)

    fit_reduced = fit_mdypl(data_reduced, alpha=0.7)
    fit_full = fit_mdypl(data_full, alpha=0.8)  # different alpha

    with pytest.raises(ValueError, match="alpha"):
        penalised_lrt(fit_reduced, fit_full)


@pytest.mark.functional
def test_penalised_lrt_raises_on_response_mismatch():
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]
    y_other = Y_DATA.copy()
    y_other[0] = 1 - y_other[0]  # flip one observation

    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA)
    data_full = prepare_mdypl_data(x=x_full, y=y_other)

    fit_reduced = fit_mdypl(data_reduced, alpha=0.8)
    fit_full = fit_mdypl(data_full, alpha=0.8)

    with pytest.raises(ValueError, match="response"):
        penalised_lrt(fit_reduced, fit_full)


@pytest.mark.functional
def test_penalised_lrt_raises_on_weights_mismatch():
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]
    weights_a = np.ones(len(Y_DATA))
    weights_b = np.full(len(Y_DATA), 2.0)

    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA, weights=weights_a)
    data_full = prepare_mdypl_data(x=x_full, y=Y_DATA, weights=weights_b)

    fit_reduced = fit_mdypl(data_reduced, alpha=0.8)
    fit_full = fit_mdypl(data_full, alpha=0.8)

    with pytest.raises(ValueError, match="weights"):
        penalised_lrt(fit_reduced, fit_full)


@pytest.mark.functional
def test_penalised_lrt_raises_on_identical_rank():
    data_a = prepare_mdypl_data(x=X_DATA, y=Y_DATA)
    data_b = prepare_mdypl_data(x=X_DATA, y=Y_DATA)  # same shape, same rank

    fit_a = fit_mdypl(data_a, alpha=0.8)
    fit_b = fit_mdypl(data_b, alpha=0.8)

    with pytest.raises(ValueError, match="rank"):
        penalised_lrt(fit_a, fit_b)


@pytest.mark.functional
def test_penalised_lrt_order_invariant():
    """full/reduced assignment should be automatic based on rank,
    regardless of argument order."""
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]
    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA)
    data_full = prepare_mdypl_data(x=x_full, y=Y_DATA)
    fit_reduced = fit_mdypl(data_reduced, alpha=0.8)
    fit_full = fit_mdypl(data_full, alpha=0.8)

    result_a = penalised_lrt(fit_reduced, fit_full)
    result_b = penalised_lrt(fit_full, fit_reduced)  # swapped order

    assert_allclose(result_a.statistic, result_b.statistic)
    assert result_a.rank_full == result_b.rank_full == fit_full.rank
    assert result_a.rank_restricted == result_b.rank_restricted == fit_reduced.rank


@pytest.mark.functional
def test_penalised_lrt_warns_and_clips_on_negative_deviance_difference():
    """If the reduced model's deviance is (spuriously) lower than the full
    model's, the statistic should be clipped to 0 with a RuntimeWarning,
    not allowed to go negative."""
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]
    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA)
    data_full = prepare_mdypl_data(x=x_full, y=Y_DATA)
    fit_reduced = fit_mdypl(data_reduced, alpha=0.8)
    fit_full = fit_mdypl(data_full, alpha=0.8)

    # Force an artificially inverted deviance to trigger the negative-difference guard
    fit_full_bad = dataclasses.replace(
        fit_full, deviance_adj=fit_reduced.deviance_adj + 1.0
    )

    with pytest.warns(RuntimeWarning, match="negative"):
        result = penalised_lrt(fit_reduced, fit_full_bad)

    assert result.statistic == 0.0


@pytest.mark.brglm2
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

    assert_allclose(fit_reduced.deviance_adj, 3.062321, rtol=1e-5)
    assert_allclose(fit_full.deviance_adj, 2.956846, rtol=1e-5)
    assert fit_reduced.rank == 2
    assert fit_full.rank == 3

    result = penalised_lrt(
        fit_reduced,
        fit_full,
        hd_correction=False,
        solve_se_kwargs={"warn_interpolator_issues": False},
    )

    expected_statistic = 0.105474361567877
    expected_df = 1
    expected_p_value = 0.745356539561018

    assert_allclose(result.statistic, expected_statistic, rtol=1e-6)
    assert result.df == expected_df
    assert_allclose(result.p_value, expected_p_value, rtol=1e-6)


@pytest.mark.brglm2
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

    result = penalised_lrt(
        fit_reduced,
        fit_full,
        hd_correction=True,
        solve_se_kwargs={"warn_interpolator_issues": False},
    )

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


@pytest.mark.functional
def test_penalised_lrt_results_str_contains_key_fields():
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]
    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA)
    data_full = prepare_mdypl_data(x=x_full, y=Y_DATA)
    fit_reduced = fit_mdypl(data_reduced, alpha=0.8)
    fit_full = fit_mdypl(data_full, alpha=0.8)

    result = penalised_lrt(fit_reduced, fit_full)
    text = str(result)

    assert "Penalised Likelihood Ratio Test" in text
    assert "Pr(>Chi)" in text


@pytest.mark.functional
def test_penalised_lrt_results_repr_is_concise():
    x_full = X_DATA
    x_reduced = X_DATA[:, [0, 1]]
    data_reduced = prepare_mdypl_data(x=x_reduced, y=Y_DATA)
    data_full = prepare_mdypl_data(x=x_full, y=Y_DATA)
    fit_reduced = fit_mdypl(data_reduced, alpha=0.8)
    fit_full = fit_mdypl(data_full, alpha=0.8)

    result = penalised_lrt(fit_reduced, fit_full)
    assert "PenalisedLRTResults" in repr(result)
