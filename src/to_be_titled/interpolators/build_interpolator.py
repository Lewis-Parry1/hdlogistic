"""
This module constructs a three cubic interpolators on a 100x100 grid
of true values of `mu`, `b` and `sigma`.
"""

from __future__ import annotations
from typing import cast

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from to_be_titled.types import FloatArray

GRID_PATH = Path(__file__).resolve().parent / "true_reference_param_grid.npz"

FieldFunc = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True)
class RgiCubicInterpolators:
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

        return cast(FloatArray, out[0])


def _rgi_field(
    kappas: FloatArray, gammas: FloatArray, param_values: FloatArray
) -> FieldFunc:
    """Fits an RGI given kappa and gamma arrays with corresponding
    2D array of specified parameter value.
    """
    rgi = RegularGridInterpolator(
        points=(kappas, gammas),
        values=param_values,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )

    def field(kappas: FloatArray, gammas: FloatArray) -> FloatArray:
        points = np.stack([np.asarray(kappas), np.asarray(gammas)], axis=-1)
        return np.asarray(rgi(points), dtype=np.float64)

    return field


@lru_cache(maxsize=1)
def _build_rgi_cubic_interpolator() -> RgiCubicInterpolators:
    """
    Build once and cache RegularGridInterpolators for `mu`, `b` and `sigma`
    over (kappa, gamma) grid of true values. Subsquent calls returns
    the cached instance without rebuilding the interpolators.
    """
    with np.load(GRID_PATH) as grid:
        kappas = grid["kappa"]
        gammas = grid["gamma"]

        mu_values = grid["mu"]
        b_values = grid["b"]
        sigma_values = grid["sigma"]

    return RgiCubicInterpolators(
        kappa_arr=kappas,
        gamma_arr=gammas,
        mu=_rgi_field(kappas, gammas, mu_values),
        b=_rgi_field(kappas, gammas, b_values),
        sigma=_rgi_field(kappas, gammas, sigma_values),
    )
