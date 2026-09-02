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
    logistic_aic,
    logistic_bic, 
    compute_deviance, 
    compute_deviance_residuals, 
    compute_pearson_residuals
)
from to_be_titled.solvers import solve_state_equation
from to_be_titled.types import FloatArray


@dataclass(frozen=True)
class HDDiagnostics:
    """Parameters for high-dimensional asymptotic corrections"""

    kappa: float
    signal_strength: float
    nu_sloe: float
    se_params: FloatArray
    opt_chain: str


@dataclass(frozen=True)
class MDYPLSummary:
    params: FloatArray
    bse: FloatArray
    zvalues: FloatArray
    pvalues: FloatArray

    linear_predictors: FloatArray
    fitted_probs: FloatArray

    nobs_eff: float

    deviance: float
    null_deviance: float 
    resid_deviance: FloatArray
    resid_pearson: FloatArray
    aic: float
    bic: float

    hd_diagnostics: HDDiagnostics | None = None

    @property
    def has_hd_correction(self) -> bool:
        """Check whether high-dimensional asymptotic corrections were applied."""
        return self.hd_diagnostics is not None

    @property
    def fittedvalues(self) -> FloatArray:
        return self.linear_predictors

    @property
    def fittedprobs(self) -> FloatArray:
        return self.fitted_probs


def summary(
    result: MDYPLResults,
    start: FloatArray | None = None,
    solve_se_kwargs: dict[str, Any] | None = None,
    high_dimensional_correction: bool = True,
) -> MDYPLSummary:
    
    x = result.x
    nobs_eff = result.nobs_eff
    eps = 1e-15
    params = result.params.copy()

    if not high_dimensional_correction:
        cov = np.asarray(result.cov_params, dtype=np.float64)
        bse = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        zvalues = params / bse
        pvalues = 2.0 * norm.cdf(-np.abs(zvalues))

        # No need to recompute 
        linear_predictors = result.linear_predictors
        fitted_probs = result.fitted_probs
        deviance = result.deviance
        null_deviance = result.null_deviance
        resid_deviance = result.resid_deviance
        resid_pearson = result.resid_pearson
        aic = result.aic
        bic = result.bic

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
        mu_star = se_params.solution.mu
        sigma_star = se_params.solution.sigma

        no_int = np.ones(len(params), dtype=bool)
        if has_intercept and intercept_idx is not None:
            no_int[intercept_idx] = False

        params[no_int] = params[no_int] / mu_star

        taus = compute_taus(x, intercept_idx)
        bse = np.empty_like(params)
        bse[no_int] = sigma_star / (np.sqrt(nobs_eff) * taus * mu_star)

        zvalues = np.empty_like(params)
        pvalues = np.empty_like(params)
        zvalues[no_int] = params[no_int] / bse[no_int]
        pvalues[no_int] = 2.0 * norm.cdf(-np.abs(zvalues[no_int]))

        if has_intercept and intercept_idx is not None:
            params[intercept_idx] = se_params.solution.intercept_estimate
            bse[intercept_idx] = np.nan
            zvalues[intercept_idx] = np.nan
            pvalues[intercept_idx] = np.nan

        # Compute gamma ^2 (estimate)
        signal_strength = float(
            derive_gamma_from_nu(kappa, nu_sloe, sigma_star, mu_star) ** 2
        )

        linear_predictors = x @ params
        if result.offset is not None:
            linear_predictors = linear_predictors + result.offset
        fitted_probs = expit(linear_predictors)

        # Null deviance does not need to be recomputed as null model fit on y_raw
        # Deviance, residual deviance and pearson residuals built on y_raw
        deviance = compute_deviance(result.y_raw, fitted_probs, result.weights, eps)
        resid_deviance = compute_deviance_residuals(result.y_raw, fitted_probs, result.weights, eps)
        resid_pearson = compute_pearson_residuals(result.y_raw, fitted_probs, result.weights, eps)

        aic = logistic_aic(result.y_adj, fitted_probs, result.weights, result.rank, eps)
        bic = logistic_bic(result.y_adj, fitted_probs, result.weights, result.rank, eps)

        hd_diagnostics = HDDiagnostics(
            kappa=kappa,
            signal_strength=signal_strength, # Note: This is gamma^2 not gamma
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
        null_deviance=result.null_deviance,
        resid_deviance=resid_deviance,
        resid_pearson=resid_pearson,
        aic=aic,
        bic=bic,
        hd_diagnostics=hd_diagnostics,
    )
