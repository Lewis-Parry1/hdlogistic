from typing import Any, cast

import numpy as np
from scipy.stats import norm
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable

from to_be_titled.inference import (
    _derive_gamma_from_nu,
    compute_sloe,
    compute_taus,
    logist_aic,
)
from to_be_titled.solvers import solve_state_equation
from to_be_titled.types import FloatArray, MDYPLResults


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

    def __getattr__(self, name: str) -> Any:
        return getattr(self.results, name)

    def __str__(self) -> str:
        return self._build_summary_str()

    def __repr__(self) -> str:
        return self._build_summary_str()

    def _apply_hd_correction(self, solve_se_kwargs: dict[str, Any]) -> None:
        res = self.results

        # Effective sample size; if freq_weights were not defined
        # then ESS = number of observations (as usual)
        self.nobs_eff = float(np.sum(res.prior_weights))

        has_intercept = res.has_intercept
        intercept_idx = res.intercept_idx

        p = len(self.params) - int(has_intercept)

        theta_hat = res.intercept

        nu_sloe = compute_sloe(
            res.y_adj, res.linear_predictors, res.fitted_probs, res.leverages
        )

        solve_se_kwargs = dict(solve_se_kwargs)
        solve_se_kwargs.update(
            kappa=p / self.nobs_eff,
            signal_strength=nu_sloe,
            alpha=res.alpha,
            intercept=theta_hat,
            corrupted=True,
        )

        se_params, opt_chain = solve_state_equation(**solve_se_kwargs)
        self.opt_chain = opt_chain  # chain of optimisation strategies needed to solve

        x = cast(FloatArray, res.model.exog)
        taus = compute_taus(x, res.intercept_idx)

        no_int = np.ones(len(self.params), dtype=bool)
        if has_intercept:
            no_int[intercept_idx] = False

        # do not update intercept (yet)
        self.params[no_int] = self.params[no_int] / se_params.solution.mu

        # rescale standard errors
        self.stand_errors[no_int] = se_params.solution.sigma / (
            np.sqrt(self.nobs_eff) * taus * se_params.solution.mu
        )

        # recompute t-stat and corresponding pvalue
        self.tvalues[no_int] = self.params[no_int] / self.stand_errors[no_int]
        self.pvalues[no_int] = 2 * norm.cdf(-np.abs(self.tvalues[no_int]))

        if res.has_intercept:
            # intercept updated to iota/theta
            self.params[intercept_idx] = se_params.solution.intercept_estimate
            # no current literature telling us how to update these
            self.stand_errors[intercept_idx] = np.nan
            self.tvalues[intercept_idx] = np.nan
            self.pvalues[intercept_idx] = np.nan

        # Save updated attributes
        self.kappa = p / self.nobs_eff

        # Retrive gamma^2 from nu
        self.signal_strength = (
            _derive_gamma_from_nu(
                self.kappa, nu_sloe, se_params.solution.sigma, se_params.solution.mu
            )
            ** 2
        )
        self.nu_sloe = nu_sloe
        self.se_params = se_params.solution.to_array()

        family = res.model.family

        self.linear_predictors = x @ self.params
        # essentially just applying sigmoid function, but keeping
        # us using family properties in statsmodels
        self.fitted_probs = family.link.inverse(self.linear_predictors)

        # Use the resid deviance and deviance formula for Binimial family
        self.deviance_resid = family.resid_dev(
            res.y_raw, self.fitted_probs, res.prior_weights
        )
        self.deviance = family.deviance(res.y_raw, self.fitted_probs, res.prior_weights)

        # Compute AIC with pseudo-responses, usual AIC wont work as y_i in (0,1)
        self.aic = (
            logist_aic(res.y_adj, self.fitted_probs, res.prior_weights) + 2.0 * res.rank
        )

    def _build_coef_table(self, param_names: list[str]) -> SimpleTable:
        z = norm.ppf(0.975)
        lower = self.params - z * self.stand_errors
        upper = self.params + z * self.stand_errors

        headers = ["", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"]
        rows = []
        for name, p, se, t, pv, lo, hi in zip(
            param_names,
            self.params,
            self.stand_errors,
            self.tvalues,
            self.pvalues,
            lower,
            upper,
        ):
            if np.isnan(se):
                rows.append([name, f"{p:.4f}", "nan", "nan", "nan", "nan", "nan"])
            else:
                rows.append(
                    [
                        name,
                        f"{p:.4f}",
                        f"{se:.3f}",
                        f"{t:.3f}",
                        f"{pv:.3f}",
                        f"{lo:.3f}",
                        f"{hi:.3f}",
                    ]
                )

        return SimpleTable(rows, headers=headers, title=None)

    def _build_summary_str(self) -> str:
        """Builds the statsmodels-style summary as a string."""
        if not self.hd_correction:
            return str(self.fitted_model.summary())

        res = self.results
        model = res.model
        param_names = list(
            getattr(model, "exog_names", None)
            or [f"x{i}" for i in range(len(self.params))]
        )

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
        smry.tables.append(self._build_coef_table(param_names))

        footer = (
            "\nHigh Dimensionality Correction applied"
            f"\nDimensionality parameter (kappa)   = {round(float(self.kappa), 3)}"
            "\nEstimated signal strength (gamma^2) ="
            f"{round(float(self.signal_strength), 3)}"
            f"\nState evolution parameters (mu, b, sigma,"
            f"(theta/iota)):{self.se_params}"
        )

        return str(smry) + footer
