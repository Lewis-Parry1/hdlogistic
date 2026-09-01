from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import chi2

from to_be_titled.summary import summary
from to_be_titled.types import FloatArray

if TYPE_CHECKING:
    from to_be_titled.estimation import MDYPLResults


@dataclass(frozen=True)
class PenalisedLRTResults:
    statistic: float
    df: int
    p_value: float
    deviance_full: float
    deviance_restricted: float
    rank_full: int
    rank_restricted: int
    hd_correction: bool
    kappa: float | None = None
    se_params: FloatArray | None = None
    signal_strength: float | None = None


def penalised_lrt(
    fit_1: MDYPLResults,
    fit_2: MDYPLResults,
    *,
    hd_correction: bool = False,
    solve_se_kwargs: dict[str, Any] | None = None,
) -> PenalisedLRTResults:
    """Compute the penalised likelihood ratio test between two nested MDYPL fits.

    Whichever model has the larger rank is automatically treated as the full model
    When `hd_correction=True`, the state evolution equations are solved on the full
    model to rescale the raw deviance drop by (b / (kappa * sigma^2)).

    Parameters
    ----------
    fit_1 : MDYPLResults
        First fitted model result
    fit_2 : MDYPLResults
        Second fitted model result (order does not matter)
    hd_correction : bool, default=False
        Whether to rescale the test statistic using high-dimensional asymptotics.
    solve_se_kwargs : dict[str, Any] | None, default=None
        Keyword arguments forwarded to `solve_state_equation` via `summary`[cite: 8].

    Returns
    -------
    PenalisedLRTResults
        Dataclass containing test statistic, degrees of freedom, p-value, and
        HD diagnostics[cite: 8, 9].
    """
    if not np.isclose(fit_1.alpha, fit_2.alpha):
        raise ValueError(
            f"`alpha` must be the same across the two fitted models"
            f"({fit_1.alpha} vs {fit_2.alpha})"
        )

    if not np.array_equal(fit_1.y_raw, fit_2.y_raw):
        raise ValueError("Models must be fit on the exact same response vector `y_raw`")

    if fit_1.rank > fit_2.rank:
        full, reduced = fit_1, fit_2
    elif fit_2.rank > fit_1.rank:
        full, reduced = fit_2, fit_1
    else:
        raise ValueError(
            f"Both models have identical rank ({fit_1.rank}). "
            "Nested models must have differing ranks to perform likelihood ratio test."
        )

    df_difference = full.rank - reduced.rank

    lrt_stat = max(0.0, float(reduced.deviance - full.deviance))

    kappa: float | None = None
    se_params: FloatArray | None = None
    signal_strength: float | None = None

    if hd_correction:
        full_summary = summary(
            full,
            solve_se_kwargs=solve_se_kwargs,
            high_dimensional_correction=True,
        )

        if full_summary.hd_diagnostics is None:
            raise RuntimeError(
                "Failed to extract high-dimensional diagnostics from full model"
            )

        hd = full_summary.hd_diagnostics
        kappa = hd.kappa
        se_params = hd.se_params
        signal_strength = hd.signal_strength

        b_hat = float(se_params[1])
        sigma_hat = float(se_params[2])

        lrt_stat = (lrt_stat * b_hat) / (kappa * (sigma_hat**2))
        lrt_stat = max(0.0, float(lrt_stat))

    p_value = float(chi2.sf(lrt_stat, df_difference))

    return PenalisedLRTResults(
        statistic=lrt_stat,
        df=df_difference,
        p_value=p_value,
        deviance_full=float(full.deviance),
        deviance_restricted=float(reduced.deviance),
        rank_full=full.rank,
        rank_restricted=reduced.rank,
        hd_correction=hd_correction,
        kappa=kappa,
        se_params=se_params,
        signal_strength=signal_strength,
    )
