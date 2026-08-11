"""
These tests ensure that build_interpolator.py correctly constructs
cubic interpolators using a small random subsample drawn from the real
reference grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from to_be_titled.interpolators.build_interpolator import (
    _build_rgi_cubic_interpolator,
)

GRID_PATH = (
    Path(__file__).resolve().parent.parent
    / "to_be_titled"
    / "interpolators"
    / "true_reference_param_grid.npz"
)


class TestBuildRgiCubicInterpolator:
    def test_is_cached_returns_same_instance(self) -> None:
        first = _build_rgi_cubic_interpolator()
        second = _build_rgi_cubic_interpolator()

        assert first is second

    def test_loads_correct_kappa_gamma(self) -> None:
        interp = _build_rgi_cubic_interpolator()
        with np.load(GRID_PATH) as grid:
            np.testing.assert_array_equal(interp.kappa_arr, grid["kappa"])
            np.testing.assert_array_equal(interp.gamma_arr, grid["gamma"])

    def test_evaluate_output_shape(self) -> None:
        interp = _build_rgi_cubic_interpolator()
        kappa_in = float(interp.kappa_arr[len(interp.kappa_arr) // 2])
        gamma_in = float(interp.gamma_arr[len(interp.gamma_arr) // 2])

        out = interp.evaluate(kappa=kappa_in, gamma=gamma_in)

        assert out.shape == (3,)
        assert out.dtype == np.float64

    def test_evaluate_recovers_exact_value(self) -> None:
        interp = _build_rgi_cubic_interpolator()

        with np.load(GRID_PATH) as grid:
            # off - diaganol points
            kappa_idx = len(interp.kappa_arr) // 2 + 1
            gamma_idx = len(interp.gamma_arr) // 2 - 1

            kappa_val = float(grid["kappa"][kappa_idx])
            gamma_val = float(grid["gamma"][gamma_idx])

            expected_value = np.array(
                [
                    grid["mu"][kappa_idx, gamma_idx],
                    grid["b"][kappa_idx, gamma_idx],
                    grid["sigma"][kappa_idx, gamma_idx],
                ]
            )

        result = interp.evaluate(kappa=kappa_val, gamma=gamma_val)
        np.testing.assert_allclose(result, expected_value, rtol=1e-6)

    def test_evaluate_clips_above(self) -> None:
        interp = _build_rgi_cubic_interpolator()

        kappa_max = float(interp.kappa_arr[-1])

        kappa_beyond_max = kappa_max + 5.0  # non-sensical kappa, but will be clipped

        gamma_mid = float(interp.gamma_arr[len(interp.gamma_arr) // 2])

        at_boundary_pt = interp.evaluate(kappa_max, gamma_mid)
        beyond_boundary_pt = interp.evaluate(kappa_beyond_max, gamma_mid)

        np.testing.assert_allclose(beyond_boundary_pt, at_boundary_pt, rtol=1e-6)

    def test_evaluate_clip_gamma_below(self) -> None:
        interp = _build_rgi_cubic_interpolator()

        kappa_mid = float(interp.kappa_arr[len(interp.kappa_arr) // 2])
        gamma_min = float(interp.gamma_arr[0])

        gamma_beyond_low = gamma_min - 5.0  # non-sensical gamma, but will be clipped

        at_boundary_pt = interp.evaluate(kappa_mid, gamma_min)
        beyond_boundary_pt = interp.evaluate(kappa_mid, gamma_beyond_low)

        np.testing.assert_allclose(at_boundary_pt, beyond_boundary_pt, rtol=1e-6)
