from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from to_be_titled.estimation import MDYPLResults
from to_be_titled.inference import (
    compute_aic,
    compute_deviance,
    compute_deviance_residuals,
    compute_likelihood,
    compute_sloe,
    compute_taus,
    derive_gamma_from_nu,
)
from to_be_titled.solvers import solve_state_equation
from to_be_titled.solvers.state_equations_solver import ConvergenceCode
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
        Leave-one-out estimate of the linear predictor variance (SLOE).
    se_params : FloatArray
        Array containing the converged state evolution parameters `(mu, b, sigma)`,
        plus the population intercept estimate `theta_0` as a 4th element if
        the model has an intercept.
    func_value : FloatArray
        The vector of residuals when evaluating the state evolution equations at
        the returned solution, used to assess solver convergence (all entries
        should be close to zero at a valid solution).
    opt_chain : ConvergenceCode
        Convergence status of the state evolution equation solver. One of:
        `DID_NOT_CONVERGE` (0) if the solver failed to converge,
        `CONVERGED_FIRST_TRY` (1) if the initial solve converged directly, or
        `CONVERGED_AFTER_INIT` (2) if convergence required a fallback/re-initialisation
        step.
    """

    kappa: float
    signal_strength: float
    nu_sloe: float
    se_params: FloatArray
    func_value: FloatArray
    opt_chain: ConvergenceCode


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
    deviance_adj : float
        Deviance evaluated on adjusted responses (penalized likelihood).
        Consistent with the objective the model was fit on; used internally
        for PLRT-type comparisons.
    deviance_raw : float
        Deviance evaluated on raw binary responses. Interpretable goodness-of-fit
        to the observed data; not comparable across nested models in the PLRT sense.
    null_deviance_adj : float
        Total deviance of the null model. Computed as twice the
        the differene between the full model and the null model.
        Deviance is returned using the DY prior penalised likelihood
        ie. the penalised deviance (uses the adjusted responses).
    null_deviance_raw : float
        Total deviance of the null model. Computed as twice the
        the differene between the full model and the null model.
        Deviance is returned using the unpenalised likelihood
        ie. the unpenalised deviance (uses the binary responses).
    resid_deviance_adj : FloatArray
        Per-observation deviance residuals evaluated on adjusted responses,
        at the (possibly rescaled) fitted probabilities.
    resid_deviance_raw : FloatArray
        Per-observation deviance residuals evaluated on raw binary responses,
        at the (possibly rescaled) fitted probabilities.
    aic : float
        Always evaluated on adjusted responses (penalized likelihood), regardless
        of `high_dimensional_correction`.
    llf : float
        Total log-likelihood of the fitted model, always evaluated on the
        adjusted response `y_adj` (the DY prior penalised likelihood).
        Consistent with `aic`, since AIC is derived from this value.
        Reuses `result.llf` unchanged if `hd_correction=False`; recomputed
        against the rescaled `fitted_probs` if `hd_correction=True`.
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

    deviance_adj: float
    deviance_raw: float
    null_deviance_adj: float
    null_deviance_raw: float
    resid_deviance_adj: FloatArray
    resid_deviance_raw: FloatArray

    aic: float
    llf: float

    hd_diagnostics: HDDiagnostics | None = None

    @property
    def has_hd_correction(self) -> bool:
        """Check whether high-dimensional asymptotic corrections were applied."""
        return self.hd_diagnostics is not None


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

    # Not hd_correction dependent
    null_deviance_raw = compute_deviance(
        result.y_raw, result.null_fitted_probs, result.weights, eps
    )

    if not high_dimensional_correction:
        cov = np.asarray(result.cov_params, dtype=np.float64)
        bse = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        zvalues = params / bse
        pvalues = 2.0 * norm.cdf(-np.abs(zvalues))

        # No need to recompute
        linear_predictors = result.linear_predictors
        fitted_probs = result.fitted_probs

        deviance_adj = result.deviance_adj
        deviance_raw = compute_deviance(
            result.y_raw, result.fitted_probs, result.weights, eps
        )

        resid_deviance_adj = result.resid_deviance_adj
        resid_deviance_raw = compute_deviance_residuals(
            result.y_raw, result.fitted_probs, result.weights, eps
        )

        llf = result.llf
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

        se_params, convergence_code = solve_state_equation(**solver_kwargs)

        func_value = se_params.func_value
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

        signal_strength = float(
            derive_gamma_from_nu(kappa, nu_sloe, sigma_star, mu_star) ** 2
        )

        linear_predictors = x @ params
        if result.offset is not None:
            linear_predictors = linear_predictors + result.offset
        fitted_probs = expit(linear_predictors)

        deviance_raw = compute_deviance(result.y_raw, fitted_probs, result.weights, eps)
        deviance_adj = compute_deviance(result.y_adj, fitted_probs, result.weights, eps)

        resid_deviance_raw = compute_deviance_residuals(
            result.y_raw, fitted_probs, result.weights, eps
        )
        resid_deviance_adj = compute_deviance_residuals(
            result.y_adj, fitted_probs, result.weights, eps
        )
        llf = compute_likelihood(
            result.y_adj, result.weights, fitted_probs, log=True, eps=eps
        )
        aic = compute_aic(llf, result.rank)

        hd_diagnostics = HDDiagnostics(
            kappa=kappa,
            signal_strength=signal_strength,  # Note: This is gamma^2 not gamma
            nu_sloe=nu_sloe,
            se_params=se_params.solution.to_array(),
            opt_chain=convergence_code,
            func_value=func_value,
        )

    return MDYPLSummary(
        params=params,
        bse=bse,
        zvalues=zvalues,
        pvalues=pvalues,
        linear_predictors=linear_predictors,
        fitted_probs=fitted_probs,
        nobs_eff=nobs_eff,
        null_deviance_adj=result.null_deviance_adj,
        null_deviance_raw=null_deviance_raw,
        deviance_adj=deviance_adj,
        deviance_raw=deviance_raw,
        resid_deviance_raw=resid_deviance_raw,
        resid_deviance_adj=resid_deviance_adj,
        aic=aic,
        llf=llf,
        hd_diagnostics=hd_diagnostics,
    )
