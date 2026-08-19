import numpy as np  

from scipy.stats import norm
from typing import cast

from to_be_titled.inference import compute_sloe, compute_taus, _derive_gamma_from_nu, logist_aic
from to_be_titled.solvers import solve_state_equation
from to_be_titled.types import (
    FloatArray, MDYPLResults
)


class MDYPLSummary:
    def __init__(
            self, 
            results: MDYPLResults, 
            hd_correction: bool = True, 
            solve_se_kwargs: dict | None = None, 
    ): 
        self.results = results
        self.fitted_model = results._results
        self.hd_correction = hd_correction

        self.params = results.params
        self.stand_errors = results.bse
        self.tvalues = results.tvalues
        self.pvalues = results.pvalues 

        if hd_correction: 
            self._apply_hd_correction(solve_se_kwargs or {})

        print(self._summary())

    def __getattr__(self, name):
        return getattr(self.results, name)

    def _apply_hd_correction(self, solve_se_kwargs: dict): 
        res = self.results

        # Effective sample size; if freq_weights were not defined 
        # then ESS = number of observations 
        nobs_eff = float(np.sum(res.model.weights)) 

        has_intercept = res.has_intercept
        intercept_idx = res.intercept_idx

        p = len(self.params) - int(has_intercept)

        theta_hat = res.intercept 

        nu_sloe = compute_sloe(res)

        # Update user's supplied state equation solver 
        # arguments 
        solve_se_kwargs = dict(solve_se_kwargs) 
        solve_se_kwargs.update(
            kappa = p / nobs_eff,
            ss = nu_sloe, 
            alpha = res.alpha, 
            intercept = theta_hat, 
            corrupted = True,
        )

        se_params, opt_chain = solve_state_equation(**solve_se_kwargs)
        self.opt_chain = opt_chain

        taus = compute_taus(res)

        no_int = np.ones(len(self.params), dtype = bool)
        if has_intercept: 
            no_int[intercept_idx] = False 

        self.params[no_int] = self.params[no_int] / se_params.solution.mu

        self.stand_errors[no_int] = (se_params.solution.sigma / 
                                (np.sqrt(nobs_eff) * taus * se_params.solution.mu))

        self.tvalues = self.params / self.stand_errors
        self.pvalues = 2 * norm.sf(- np.abs(self.tvalues))

        if res.has_intercept: 
            self.params[intercept_idx] = se_params.solution.intercept_estimate
            self.stand_errors[intercept_idx] = np.nan
            self.tvalues[intercept_idx] = np.nan
            self.pvalues[intercept_idx] = np.nan

        self.kappa = p / nobs_eff
        self.signal_strength = (_derive_gamma_from_nu(self.kappa, 
                                                     nu_sloe, 
                                                     se_params.solution.sigma, 
                                                     se_params.solution.mu) **2)
        self.nu_sloe = nu_sloe
        self.se_params = se_params.solution.to_array()

        family = res.model.family
        self.linear_predictors = cast(FloatArray, res.model.exog) @ self.params
        self.fitted_probs = family.link.inverse(self.linear_predictors)

        # Null deviance does not need to be updated as intercept is not rescaled 
        # Recompute deviance with new coeffcient estimates 

        self.deviance_resid = family.resid_dev(res.y_adj, self.fitted_probs, var_weights=res.weights)
        self.deviance = family.deviance(res.y_adj, self.fitted_probs, var_weights=res.weights)
        self.aic = (logist_aic(res.y_adj, self.fitted_probs, res.weights) + 2.0 * res.rank) 

        
    def _summary(self) -> None:
        """Prints statsmodels style summary. 
        """
        if not self.hd_correction: 
            print(self.fitted_model.summary())
            return
        else: 


#TODO: Confidence intervals

