from typing import Any, cast

import numpy as np
from scipy.stats import norm
from statsmodels.iolib.summary import Summary

from to_be_titled.inference import (
    _derive_gamma_from_nu,
    compute_sloe,
    compute_taus,
    logist_aic,
)
from to_be_titled.solvers import solve_state_equation
from to_be_titled.types import FloatArray, MDYPLResults


class _ParamsView:
    """ """

    def __init__(
        self,
        params: FloatArray,
        bse: FloatArray,
        tvalues: FloatArray,
        pvalues: FloatArray,
    ):
        self.params = np.asarray(params)
        self.bse = np.asarray(bse)
        self.tvalues = np.asarray(tvalues)
        self.pvalues = np.asarray(pvalues)

    def conf_int(self, alpha: float = 0.05) -> FloatArray:
        z = norm.ppf(1.0 - alpha / 2.0)
        lower = self.params - z * self.bse
        upper = self.params + z * self.bse
        return np.column_stack([lower, upper])


class MDYPLSummary:
    def __init__(
        self,
        results: MDYPLResults,
        hd_correction: bool = True,
        solve_se_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.results = results
        self.fitted_model = results._results
        self.hd_correction = hd_correction

        self.params = results.params.copy()
        self.stand_errors = results.bse.copy()
        self.tvalues = results.tvalues.copy()
        self.pvalues = results.pvalues.copy()

        if hd_correction:
            self._apply_hd_correction(solve_se_kwargs or {})

        self._summary()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.results, name)

    def _apply_hd_correction(self, solve_se_kwargs: dict[str, Any]) -> None:
        res = self.results

        # Effective sample size; if freq_weights were not defined
        # then ESS = number of observations
        self.nobs_eff = float(np.sum(res.prior_weights))

        has_intercept = res.has_intercept
        intercept_idx = res.intercept_idx

        p = len(self.params) - int(has_intercept)

        theta_hat = res.intercept

        nu_sloe = compute_sloe(
            res.y_adj, res.linear_predictors, res.fitted_probs, res.leverages
        )

        # Update user's supplied state equation solver
        # arguments
        solve_se_kwargs = dict(solve_se_kwargs)
        solve_se_kwargs.update(
            kappa=p / self.nobs_eff,
            ss=nu_sloe,
            alpha=res.alpha,
            intercept=theta_hat,
            corrupted=True,
        )

        se_params, opt_chain = solve_state_equation(**solve_se_kwargs)
        self.opt_chain = opt_chain

        taus = compute_taus(res)

        no_int = np.ones(len(self.params), dtype=bool)
        if has_intercept:
            no_int[intercept_idx] = False

        self.params[no_int] = self.params[no_int] / se_params.solution.mu

        self.stand_errors[no_int] = se_params.solution.sigma / (
            np.sqrt(self.nobs_eff) * taus * se_params.solution.mu
        )

        self.tvalues[no_int] = self.params / self.stand_errors
        self.pvalues[no_int] = 2 * norm.cdf(-np.abs(self.tvalues))

        if res.has_intercept:
            self.params[intercept_idx] = se_params.solution.intercept_estimate
            self.stand_errors[intercept_idx] = np.nan
            self.tvalues[intercept_idx] = np.nan
            self.pvalues[intercept_idx] = np.nan

        self.kappa = p / self.nobs_eff
        self.signal_strength = (
            _derive_gamma_from_nu(
                self.kappa, nu_sloe, se_params.solution.sigma, se_params.solution.mu
            )
            ** 2
        )
        self.nu_sloe = nu_sloe
        self.se_params = se_params.solution.to_array()

        family = res.model.family
        self.linear_predictors = cast(FloatArray, res.model.exog) @ self.params
        self.fitted_probs = family.link.inverse(self.linear_predictors)

        # Null deviance does not need to be updated as intercept is not rescaled
        # Recompute deviance with new coeffcient estimates on raw y
        self.deviance_resid = family.resid_dev(
            res.y_raw, self.fitted_probs, res.prior_weights
        )
        self.deviance = family.deviance(res.y_raw, self.fitted_probs, res.prior_weights)

        self.aic = (
            logist_aic(res.y_adj, self.fitted_probs, res.prior_weights) + 2.0 * res.rank
        )

    def _summary(self) -> None:
        """Prints statsmodels style summary."""
        if not self.hd_correction:
            print(self.fitted_model.summary())
            return

        res = self.results
        model = res.model
        param_names = list(
            getattr(model, "exog_names", None)
            or [f"x{i}" for i in range(len(self.params))]
        )

        pview = _ParamsView(self.params, self.stand_errors, self.tvalues, self.pvalues)

        top_left = [
            ("Dep. Variable:", [getattr(model, "endog_names", "y")]),
            ("Model:", ["MDYPL-GLM"]),
            ("Model Family:", [model.family.__class__.__name__]),
            ("Link Function:", [model.family.link.__class__.__name__]),
            ("Method:", [model.method]),
            ("No. Iterations:", [str(res.iterations)]),
        ]

        top_right = [
            ("No. Observations:", [str(self.nobs_eff)]),
            ("Df Residuals:", [str(model.df_resid)]),
            ("Df Model:", [str(model.df_model)]),
            ("Deviance:", [f"{self.deviance:.5g}"]),
            ("AIC:", [f"{self.aic:.5g}"]),
        ]

        smry = Summary()
        smry.add_table_2cols(
            res,
            gleft=top_left,
            gright=top_right,
            yname=getattr(model, "endog_names", "y"),
            title="MDYPL Regression Results (HD-corrected)",
        )

        smry.add_table_params(
            pview, xname=param_names, alpha=0.05, use_t=False
        )  # TODO: Check use_t
        print(smry)
        print("\nHigh Dimensionality Correction applied")
        print(f"Dimensionality parameter (kappa)   = {round(float(self.kappa), 3)}")
        print(
            "Estimated signal strength (gamma^2) ="
            f"{round(float(self.signal_strength), 3)}"
        )

        print(
            f"State evolution parameters (mu, b, sigma, (theta/iota)):{self.se_params}"
        )
