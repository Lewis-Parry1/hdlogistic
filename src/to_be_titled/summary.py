import numpy as np  

from scipy.stats import norm
from typing import cast

from to_be_titled.inference import compute_sloe, compute_taus, _derive_gamma_from_nu, _logist_aic
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
        self.hd_correction = hd_correction

        self.params = np.asarray(results.params, dtype= np.float64).copy()
        self.bse = np.asarray(results.bse, dtype = np.float64).copy()
        self.tvalues = np.asarray(results.tvalues, dtype= np.float64).copy()
        self.pvalues = np.asarray(results.pvalues, dtype=np.float64).copy()
        self.alpha = results.alpha

        if hd_correction: 
            self._apply_hd_correction(solve_se_kwargs or {})


    def _apply_hd_correction(self, solve_se_kwargs: dict): 
        mdypl_res = self.results

        nobs_eff = float(np.sum(mdypl_res.model.weights)) 

        has_intercept = mdypl_res.has_intercept
        intercept_idx = mdypl_res.intercept_idx
        p = len(self.params) - int(has_intercept)

        theta_hat = mdypl_res.intercept # None if no intercept 

        nu_sloe = compute_sloe(mdypl_res)

        solve_se_kwargs = dict(solve_se_kwargs) # Don't mutate caller's dict 
        solve_se_kwargs.update(
            kappa = p / nobs_eff,
            ss = nu_sloe, 
            alpha = mdypl_res.alpha, 
            intercept = theta_hat, 
            corrupted = True,
        )

        se_params, opt_chain = solve_state_equation(**solve_se_kwargs)

        # TODO: Design matrix rank defciency check conversation needed. 
        taus = compute_taus(mdypl_res)

        no_int = np.ones(len(self.params), dtype = bool)
        if has_intercept: 
            no_int[intercept_idx] = False 

        # Rescale onto MDYPLSummary's own properties 
        self.params[no_int] = self.params[no_int] / se_params.solution.mu

        self.bse[no_int] = (se_params.solution.sigma / 
                                (np.sqrt(nobs_eff) * taus * se_params.solution.mu))

        self.tvalues = self.params / self.bse
        self.pvalues = 2 * norm.sf(- np.abs(self.tvalues))

        if mdypl_res.has_intercept: 
            self.params[intercept_idx] = se_params.solution.intercept_estimate
            self.bse[intercept_idx] = np.nan
            self.tvalues[intercept_idx] = np.nan
            self.pvalues[intercept_idx] = np.nan

        self.kappa = p / nobs_eff
        self.se_params = se_params.solution
        self.signal_strength = (_derive_gamma_from_nu(self.kappa, 
                                                     nu_sloe, 
                                                     self.se_params.sigma, 
                                                     self.se_params.mu) **2)
        self.nu_sloe = nu_sloe

        linear_predictor = cast(FloatArray, mdypl_res.model.exog) @ self.params
        fitted_probs = (mdypl_res.model.family.link.inverse(linear_predictor))
        # save these ^ ?

        # TODO: Note len(self.params) will need to be updated to rank, if we 
        # decide to drop columns which are linearly dependent in the future
        self.aic = (_logist_aic(mdypl_res.y_adj, fitted_probs, mdypl_res.model.weights)
                    + 2.0 * len(self.params)
                    )

        # Deviance
        



