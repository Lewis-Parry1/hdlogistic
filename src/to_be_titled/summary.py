#
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from to_be_titled.estimation import MDYPLResults
from to_be_titled.inference import (
    compute_sloe,
    compute_taus,
    derive_gamma_from_nu,
    logist_aic,
)
from to_be_titled.solvers import solve_state_equation
from to_be_titled.types import FloatArray


@dataclass(frozen=True)
class HDDiagnostics:
    """Inference diagnostics and parameter solutions from high-dimensional asymptotics.

    Parameters
    ----------
    kappa : float
        Aspect ratio `p / nobs_eff` (ratio of predictors to effective sample size).
    signal_strength : float
        Estimated squared signal strength parameter `gamma**2`.
    nu_sloe : float
        Surrogate leave-one-out estimate of the linear predictor variance (SLOE).
    se_params : FloatArray
        Array containing the converged state evolution parameters `(alpha, mu, sigma)`.
    opt_chain : str
        Description or identifier of the optimization solver chain used to solve
        the state evolution equations.
    """

    kappa: float
    signal_strength: float
    nu_sloe: float
    se_params: FloatArray
    opt_chain: str


@dataclass(frozen=True)
class MDYPLSummary:
    """Summary container for MDYPL model inference and diagnostics.

    Parameters
    ----------
    params : FloatArray
        Parameter estimates of shape `(p,)` (rescaled/debiased if HD correction is
        enabled).
    bse : FloatArray
        Standard errors of shape `(p,)` (adjusted via state evolution parameters
        under HD correction; NaN for intercept).
    zvalues : FloatArray
        Wald $z$-statistics of shape `(p,)`.
    pvalues : FloatArray
        Two-sided asymptotic p-values of shape `(p,)`.
    linear_predictors : FloatArray
        Linear predictors `X @ params + offset` of shape `(n,)`.
    fitted_probs : FloatArray
        Fitted probabilities of shape `(n,)`.
    nobs_eff : float
        Effective sample size (sum of weights).
    deviance : float
        Model deviance evaluated at the final parameter estimates.
    aic : float
        Akaike Information Criterion evaluated on the adjusted response.
    hd_diagnostics : HDDiagnostics | None, optional
        Diagnostics and state evolution solutions from high-dimensional asymptotics,
        by default None.
    """

    params: FloatArray
    bse: FloatArray
    zvalues: FloatArray
    pvalues: FloatArray

    linear_predictors: FloatArray
    fitted_probs: FloatArray

    nobs_eff: float
    deviance: float
    aic: float

    hd_diagnostics: HDDiagnostics | None = None

    @property
    def has_hd_correction(self) -> bool:
        """Check whether high-dimensional asymptotic corrections were applied."""
        return self.hd_diagnostics is not None

    @property
    def fittedvalues(self) -> FloatArray:
        return self.linear_predictors

    @property
    def mu(self) -> FloatArray:
        return self.fitted_probs


def summary(
    result: MDYPLResults,
    start: FloatArray | None = None,
    solve_se_kwargs: dict[str, Any] | None = None,
    high_dimensional_correction: bool = True,
) -> MDYPLSummary:
    """Compute inferential statistics and asymptotic standard errors for an MDYPL fit.

    Parameters
    ----------
    result : MDYPLResults
        Fitted model results container returned by `fit_mdypl`.
    start : FloatArray | None, optional
        Initial starting values `(alpha, b, sigma)` for the state evolution
        equation solver, by default None.
    solve_se_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments forwarded to `solve_state_equation`,
        by default None.
    high_dimensional_correction : bool, optional
        Whether to adjust parameter estimates, standard errors, and p-values using
        high-dimensional asymptotics (SLOE and state evolution equations). If False,
        standard GLM covariance asymptotics are used, by default True.

    Returns
    -------
    MDYPLSummary
        Summary container storing parameter estimates, standard errors, $z$-values,
        p-values, predictions, deviance, AIC, and optional high-dimensional diagnostics.
    """
    x = result.x
    nobs_eff = result.nobs_eff
    eps = 1e-15
    params = result.params.copy()

    if not high_dimensional_correction:
        cov = np.asarray(result.cov_params, dtype=np.float64)
        bse = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        zvalues = params / bse
        pvalues = 2.0 * norm.cdf(-np.abs(zvalues))

        linear_predictors = result.linear_predictors
        fitted_probs = result.fitted_probs
        deviance = result.deviance
        aic = result.aic
        hd_diagnostics = None

    else:
        has_intercept = result.has_intercept
        intercept_idx = result.intercept_idx
        p = len(params) - int(has_intercept)
        kappa = p / nobs_eff

        nu_sloe = compute_sloe(
            result.y_adj,
            result.linear_predictors,
            result.fitted_probs,
            result.leverages,
        )

        solver_kwargs = dict(solve_se_kwargs or {})
        if start is not None:
            solver_kwargs["start"] = start

        solver_kwargs.update(
            kappa=kappa,
            signal_strength=nu_sloe,
            alpha=result.alpha,
            intercept=result.intercept,
            corrupted=True,
        )

        se_params, opt_chain = solve_state_equation(**solver_kwargs)
        mu_hat = se_params.solution.mu
        sigma_hat = se_params.solution.sigma

        no_int = np.ones(len(params), dtype=bool)
        if has_intercept and intercept_idx is not None:
            no_int[intercept_idx] = False

        params[no_int] = params[no_int] / mu_hat

        taus = compute_taus(x, intercept_idx)
        bse = np.empty_like(params)
        bse[no_int] = sigma_hat / (np.sqrt(nobs_eff) * taus * mu_hat)

        zvalues = np.empty_like(params)
        pvalues = np.empty_like(params)
        zvalues[no_int] = params[no_int] / bse[no_int]
        pvalues[no_int] = 2.0 * norm.cdf(-np.abs(zvalues[no_int]))

        if has_intercept and intercept_idx is not None:
            params[intercept_idx] = se_params.solution.intercept_estimate
            bse[intercept_idx] = np.nan
            zvalues[intercept_idx] = np.nan
            pvalues[intercept_idx] = np.nan

        signal_strength = float(
            derive_gamma_from_nu(kappa, nu_sloe, sigma_hat, mu_hat) ** 2
        )

        linear_predictors = x @ params
        if result.offset is not None:
            linear_predictors = linear_predictors + result.offset
        fitted_probs = expit(linear_predictors)

        deviance = float(
            -2.0
            * np.sum(
                result.weights
                * np.where(
                    result.y_raw == 1,
                    np.log(np.clip(fitted_probs, eps, 1.0)),
                    np.log(np.clip(1.0 - fitted_probs, eps, 1.0)),
                )
            )
        )
        aic = float(
            logist_aic(result.y_adj, fitted_probs, result.weights) + 2.0 * result.rank
        )

        hd_diagnostics = HDDiagnostics(
            kappa=kappa,
            signal_strength=signal_strength,
            nu_sloe=nu_sloe,
            se_params=se_params.solution.to_array(),
            opt_chain=opt_chain,
        )

    return MDYPLSummary(
        params=params,
        bse=bse,
        zvalues=zvalues,
        pvalues=pvalues,
        linear_predictors=linear_predictors,
        fitted_probs=fitted_probs,
        nobs_eff=nobs_eff,
        deviance=deviance,
        aic=aic,
        hd_diagnostics=hd_diagnostics,
    )
