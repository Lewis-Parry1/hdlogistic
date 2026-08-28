from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable

from to_be_titled.summary import MDYPLSummary
from to_be_titled.types import MDYPLResults


class PenalisedLRTResults:
    """Results container for `penalised_lrt`, with statsmodels-style printing."""

    def __init__(
        self,
        statistic: float,
        df: int,
        p_value: float,
        hd_correction: bool,
        kappa: float | None = None,
        se_params: NDArray | None = None,
        signal_strength: float | None = None,
    ) -> None:

        self.statistic = statistic
        self.df = df
        self.p_value = p_value
        self.hd_correction = hd_correction
        self.kappa = kappa
        self.se_params = se_params
        self.signal_strength = signal_strength

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def summary(self) -> Summary:
        smry = Summary()
        title = "Penalized Likelihood Ratio Test"
        if self.hd_correction:
            title += " (high-dimensionality corrected)"

        smry.tables.append(
            SimpleTable(
                [[f"{self.statistic:.4f}", f"{self.df}", f"{self.p_value:.4g}"]],
                headers=["LRT Statistic", "df", "Pr(>Chi)"],
                title=title,
            )
        )

        if self.hd_correction:
            assert self.se_params is not None, (
                "se_params must be set when hd_correction=True"
            )
            mu, b, sigma = self.se_params[:3]
            smry.tables.append(
                SimpleTable(
                    [
                        ["Dimensionality (kappa)", f"{self.kappa:.4f}"],
                        ["Signal strength (gamma^2)", f"{self.signal_strength:.4f}"],
                        ["State evolution mu", f"{mu:.4f}"],
                        ["State evolution b", f"{b:.4f}"],
                        ["State evolution sigma", f"{sigma:.4f}"],
                    ],
                    headers=["Diagnostic", "Value"],
                )
            )
        return smry

    def __repr__(self) -> str:
        return str(self.summary())


def penalised_lrt(
    fit_1: MDYPLResults,
    fit_2: MDYPLResults,
    hd_correction: bool = False,
    solve_se_kwargs: dict[str, Any] | None = None,
) -> PenalisedLRTResults:
    """Penalized likelihood ratio test for two nested MDYPL fits.

    `fit_1` and `fit_2` may be passed in either order — whichever has
    the larger rank is treated as the full (unrestricted) model.
    """
    if not np.isclose(fit_1.alpha, fit_2.alpha):
        raise ValueError("`alpha` must be the same across the two fitted models.")

    if not np.array_equal(fit_1.y_raw, fit_2.y_raw):
        raise ValueError("Models must be fit on the same response.")

    if fit_2.rank > fit_1.rank:
        full, reduced = fit_2, fit_1
    else:
        full, reduced = fit_1, fit_2

    dof_difference = full.rank - reduced.rank

    # = 2 * (loglike(full) - loglike(reduced))
    lrt_stat = reduced.deviance - full.deviance

    kappa = None
    se_params = None
    signal_strength = None

    if hd_correction:
        summ = MDYPLSummary(full, hd_correction=True, solve_se_kwargs=solve_se_kwargs)
        kappa = summ.kappa
        se_params = summ.se_params
        signal_strength = summ.signal_strength
        lrt_stat = (lrt_stat * se_params[1]) / (kappa * se_params[2] ** 2)

    p_value = chi2.sf(lrt_stat, dof_difference)

    return PenalisedLRTResults(
        statistic=lrt_stat,
        df=dof_difference,
        p_value=p_value,
        hd_correction=hd_correction,
        kappa=kappa,
        se_params=se_params,
        signal_strength=signal_strength,
    )
