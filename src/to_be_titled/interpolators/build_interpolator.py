"""
This module constructs three PCHIP interpolators on a 100x100 grid
of true values of `mu`, `b` and `sigma`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import cast

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from to_be_titled.types import FloatArray

GRID_PATH = files("to_be_titled") / "data" / "true_reference_param_grid.npz"

FieldFunc = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True)
class RgiPchipInterpolators:
    """Container for fitted PCHIP interpolators of `mu`, `b`, `sigma`
    over a (kappa, gamma) reference grid, used to produce warm-start
    values for `solve_state_equation`.

    Parameters
    ----------
    kappa_arr : FloatArray
        1D array of kappa grid points the interpolators were fit on.
    gamma_arr : FloatArray
        1D array of gamma grid points the interpolators were fit on.
    mu, b, sigma : FieldFunc
        Callables evaluating the interpolated `mu`/`b`/`sigma` surface
        at arbitrary (kappa, gamma) query points.
    """

    kappa_arr: FloatArray
    gamma_arr: FloatArray

    mu: FieldFunc
    b: FieldFunc
    sigma: FieldFunc

    def evaluate(self, kappa: float, gamma: float) -> FloatArray:
        """
        Interpolate paramater values given (kappa, gamma) pair.

        Parameters
        ----------
        kappa : float
            Asymptotic ratio of the columns/rows of the design matrix (p/n).
            `kappa` should be in (0,1).
        gamma : float
            Square root of the (potentially corrupted) signal strength.

        Returns
        -------
        FloatArray
            Array of interpolated point given kappa & gamma corresponding to
            `mu`, `b` and `sigma`.
        """
        # clip input such that it stays in boundaries
        kappa_c = np.asarray(np.clip(kappa, self.kappa_arr[0], self.kappa_arr[-1]))
        gamma_c = np.asarray(np.clip(gamma, self.gamma_arr[0], self.gamma_arr[-1]))

        out = np.stack(
            [
                self.mu(kappa_c, gamma_c),
                self.b(kappa_c, gamma_c),
                self.sigma(kappa_c, gamma_c),
            ],
            axis=-1,
        )

        return cast(FloatArray, np.squeeze(out, axis=0))


def _rgi_field(
    grid_kappas: FloatArray, grid_gammas: FloatArray, param_values: FloatArray
) -> FieldFunc:
    """Fits an RGI given kappa and gamma grid arrays with corresponding
    2D array of specified parameter values.
    """
    rgi = RegularGridInterpolator(
        points=(grid_kappas, grid_gammas),
        values=param_values,
        method="pchip",
        bounds_error=False,
        fill_value=None,
    )

    def field(kappa_query: FloatArray, gamma_query: FloatArray) -> FloatArray:
        points = np.stack([np.asarray(kappa_query), np.asarray(gamma_query)], axis=-1)
        return np.asarray(rgi(points), dtype=np.float64)

    return field


@lru_cache(maxsize=1)
def _build_rgi_pchip_interpolator() -> RgiPchipInterpolators:
    """
    Build once and cache RegularGridInterpolators for `mu`, `b` and `sigma`
    over (kappa, gamma) grid of true values. Subsequent calls returns
    the cached instance without rebuilding the interpolators.
    """
    if not GRID_PATH.is_file():
        raise FileNotFoundError(
            f"Reference grid for the warm-start interpolator not found at "
            f"{GRID_PATH}. This file is required when "
            f"`use_warm_start_interpolator=True` (the default). Either ensure "
            f"the package data is correctly installed, or call "
            f"`solve_state_equation(..., use_warm_start_interpolator=False)` "
            f"to skip the interpolated warm start entirely."
        )

    with GRID_PATH.open("rb") as f, np.load(f) as grid:
        kappas = grid["kappa"]
        gammas = grid["gamma"]

        mu_values = grid["mu"]
        b_values = grid["b"]
        sigma_values = grid["sigma"]

    return RgiPchipInterpolators(
        kappa_arr=kappas,
        gamma_arr=gammas,
        mu=_rgi_field(kappas, gammas, mu_values),
        b=_rgi_field(kappas, gammas, b_values),
        sigma=_rgi_field(kappas, gammas, sigma_values),
    )
