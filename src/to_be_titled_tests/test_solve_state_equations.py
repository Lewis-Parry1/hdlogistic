'''import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from to_be_titled.solve_state_equations import _se_funcs
from to_be_titled.utils import _get_hermite_roots_weights

"""

"""


def test_se_funcs_true_false_transform_agree() -> None:
    """
    If transform is true then g(log(start)) should return approximately same
    result as g(start) when trasform is False. Slight numerical difference may
    occur due to clipping inputs before exponentiaing when transform is True.
    """
    # Define arbitrary, valid coeffcients
    kappa, gamma = 0.5, 10
    # Use adaptive shrinkage as discussed in (Sterzinger and Kosmidis, 2026)
    alpha = 1 / (1 + kappa)

    gh = _get_hermite_roots_weights(200)

    g_true = _se_funcs(kappa, gamma, alpha, transform=True, gh=gh)
    g_false = _se_funcs(kappa, gamma, alpha, transform=False, gh=gh)

    start = np.array([0.5, 1.0, 1.0])
    start_log = np.asarray(np.log(start), dtype=float)

    res_transform_false = g_false(start)

    res_transform_true = g_true(start_log)

    np.testing.assert_allclose(
        res_transform_false, res_transform_true, atol=1e-10, rtol=1e-8
    )


@pytest.mark.parametrize(
    "pars",
    [
        np.array([0.5, 1e3, 1e3]),  # Well beyond clip boundary
        np.array([0.5, 250.0, 250.0]),  # At the clip boundary
        np.array([0.5, 251.0, 251.0])  # Just past the clip boundary
    ]
)
def test_se_funcs_handles_extremes_without_overflow(pars: NDArray[np.float64]) -> None:
    """
    Extreme log space inputs are clipped to ensure they lie within [-250, 250] before
    exponentiating. This should prevent g from returning overflow warnings
    before exponetiating and should stop g returning inf/nan.
    """
    kappa, gamma = 0.2, np.sqrt(0.9)
    alpha = 1 / (1 + kappa)
    gh = _get_hermite_roots_weights(50)

    g = _se_funcs(kappa, gamma, alpha, gh=gh, transform=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        result = g(pars)


    assert np.all(np.isfinite(result))
'''
