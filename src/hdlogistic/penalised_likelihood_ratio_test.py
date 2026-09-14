from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import chi2

from hdlogistic.summary import summary
from hdlogistic.types import FloatArray

if TYPE_CHECKING:
    from hdlogistic.estimation import MDYPLResults


@dataclass(frozen=True)
class PenalisedLRTResults:
    """Results container for the penalised likelihood ratio test between nested models.

    Parameters
    ----------
    statistic : float
        Likelihood ratio test statistic (rescaled by high-dimensional asymptotics
        if `hd_correction=True`).
    df : int
        Degrees of freedom for the test (`rank_full - rank_restricted`).
    p_value : float
        Asymptotic p-value computed from the chi-squared survival function.
    deviance_full : float
        Deviance of the full model. This deviance is the penalised deviance
        in the sense it uses the DY prior penalised likelihood (ie. uses adjusted
        responses).
    deviance_restricted : float
        Deviance of the nested model. This deviance is the penalised deviance
        in the sense it uses the DY prior penalised likelihood (ie. uses adjusted
        responses).
    rank_full : int
        Column rank of the full model design matrix.
    rank_restricted : int
        Column rank of the restricted model design matrix.
    hd_correction : bool
        Whether the high-dimensional state evolution rescaling was applied.
    kappa : float | None, optional
        Aspect ratio `p / n` from high-dimensional asymptotics, by default None.
    se_params : FloatArray | None, optional
        State evolution parameter solutions `(mu, b, sigma)` from the full model,
        and 'theta_0' if an intercept is included. By default None.
    signal_strength : float | None, optional
        Estimated signal strength parameter `gamma**2` from high-dimensional
        asymptotics, by default None.
    """

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

    def __str__(self) -> str:
        def sig_stars(p: float) -> str:
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            elif p < 0.1:
                return "."
            return ""

        lines = ["Penalised Likelihood Ratio Test for Nested MDYPL Models", ""]
        header = (
            f"{'':<15}{'Rank':>6}{'Deviance':>12}{'Df':>6}{'Stat':>10}{'Pr(>Chi)':>12}"
        )
        lines.append(header)
        lines.append(
            f"{'Reduced':<15}{self.rank_restricted:>6}{self.deviance_restricted:>12.4f}"
        )
        lines.append(
            f"{'Full':<15}{self.rank_full:>6}{self.deviance_full:>12.4f}"
            f"{self.df:>6}{self.statistic:>10.4f}{self.p_value:>12.4g} "
            f"{sig_stars(self.p_value)}"
        )
        lines.append("---")
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

        if self.hd_correction:
            lines.append("")
            lines.append("High-dimensionality correction applied with")
            lines.append(f"Dimensionality parameter (kappa) = {self.kappa:.4g}")
            lines.append(
                f"Estimated signal strength (gamma^2) = {self.signal_strength:.4g}"
            )
            if self.se_params is not None:
                mu_b_sigma = ", ".join(f"{v:.4g}" for v in self.se_params[:3])
                lines.append(
                    f"State evolution parameters (mu, b, sigma) = ({mu_b_sigma})"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        # keep the default dataclass repr for programmatic use / debugging
        return (
            f"PenalisedLRTResults(statistic={self.statistic!r}, df={self.df!r}, "
            f"p_value={self.p_value!r}, hd_correction={self.hd_correction!r})"
        )


def penalised_lrt(
    fit_1: MDYPLResults,
    fit_2: MDYPLResults,
    *,
    hd_correction: bool = False,
    solve_se_kwargs: dict[str, Any] | None = None,
) -> PenalisedLRTResults:
    r"""Conducts a penalised likelihood ratio test between two nested MDYPL fits.

    Whichever model has the larger rank is automatically treated as the full model.
    When `hd_correction=True`, the state evolution equations are solved using the
    (\alpha, \kappa, \gamma, \intercept) parameters from the full model,
    the resulting \sigma_*^2 and b_* are used to rescale the test statistic,
    and the p-value is computed using the \chi^2 distribution with degrees of
    freedom equal to the difference in rank between the two models.

    The LRT statistic is computed as:
        2 \Lambda = D_reduced - D_full
    The LRT statistic is then rescaled by a factor of b_*/(\kappa \sigma^2)
    when `hd_correction=True`.
    If `hd_correction=False`, the LRT statistic is not rescaled
    and is assumed to follow a \chi^2 distribution with degrees of
    freedom equal to the difference in rank between the two models.
    Then the p-value is computed as:
        p = 1 - F_{\chi^2_{df}}(2 \Lambda)

    Parameters
    ----------
    fit_1 : MDYPLResults
        First fitted model result.
    fit_2 : MDYPLResults
        Second fitted model result (order does not matter).
    hd_correction : bool, default=False
        Whether to rescale the test statistic using high-dimensional asymptotics.
    solve_se_kwargs : dict[str, Any] | None, default=None
        Keyword arguments forwarded to `solve_state_equation` via `summary`.

    Returns
    -------
    PenalisedLRTResults
        Dataclass containing test statistic, degrees of freedom, p-value, and
        HD diagnostics.
    """
    if not np.isclose(fit_1.alpha, fit_2.alpha):
        raise ValueError(
            f"`alpha` must be the same across the two fitted models"
            f"({fit_1.alpha} vs {fit_2.alpha})"
        )

    if not np.array_equal(fit_1.y_raw, fit_2.y_raw):
        raise ValueError("Models must be fit on the exact same response vector `y_raw`")

    if not np.array_equal(fit_1.weights, fit_2.weights):
        raise ValueError(
            "Models must be fit on the exact same frequency weights `weights`"
        )

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

    # LRT Stat = l_fit - l_nested = D_reduced - D_full
    deviance_difference = float(reduced.deviance_adj - full.deviance_adj)
    if (
        deviance_difference < -1e-12
    ):  # allow for floating-point noise, not real negativity
        warnings.warn(
            f"Deviance difference is negative ({deviance_difference:.3e});"
            "this may indicate"
            "non-convergence in one of the fitted models. Clipping to 0.",
            RuntimeWarning,
        )
    lrt_stat = max(0.0, deviance_difference)

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

        b_star = float(se_params[1])
        sigma_star = float(se_params[2])

        lrt_stat = (lrt_stat * b_star) / (kappa * (sigma_star**2))
        lrt_stat = max(0.0, float(lrt_stat))

    p_value = float(chi2.sf(lrt_stat, df_difference))

    return PenalisedLRTResults(
        statistic=lrt_stat,
        df=df_difference,
        p_value=p_value,
        deviance_full=float(full.deviance_adj),
        deviance_restricted=float(reduced.deviance_adj),
        rank_full=full.rank,
        rank_restricted=reduced.rank,
        hd_correction=hd_correction,
        kappa=kappa,
        signal_strength=signal_strength,
        se_params=se_params,
    )
