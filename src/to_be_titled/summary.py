import numpy as np  

from to_be_titled.inference import compute_sloe, compute_taus
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

        if hd_correction: 
            self._apply_hd_correction(solve_se_kwargs or {})

    def _apply_hd_correction(self, solve_se_kwargs: dict): 
        mdypl_res = self.results

        nobs = mdypl_res.nobs
        coef = np.asarray(mdypl_res.params, dtype=np.float64)

        fw = mdypl_res.model.freq_weights if mdypl_res.model.freq_weights is not None else np.ones(nobs)
        nobs_eff = float(fw) 

        has_intercept = mdypl_res.has_intercept
        p = len(coef) - int(has_intercept)

        theta_hat = mdypl_res.intercept # None if no intercept 

        nu_sloe = compute_sloe(mdypl_res)

        solve_se_kwargs.update(
            kappa = p / nobs_eff,
            ss = nu_sloe, 
            alpha = mdypl_res.alpha, 
            intercept = theta_hat, 
            corrupted = True,
        )

        se_params = solve_state_equation(**solve_se_kwargs)

        
        
        
        



####




def summary(
    result,
    start: FloatArray | None = None,
    high_dimensional_correction: bool = True,
) -> FloatArray:
    """Provides summary statistics from the provided model results and optionally
    applies a high-dimensional correction to the estimated coefficients.

    Parameters
    ----------
    result : DiaconisYlvisakerLogisticRegressionResult
        A dataclass containing the results of the Diaconis-Ylvisaker logistic
        regression fit.
    start : FloatArray | None, default = None
        Starting values (`mu`, `b`, `sigma`, and optionally `beta_0`) passed to the
        internal state evolution solver (`solve_state_equation`) when
        `high_dimensional_correction=True`. If `None`, defaults to preset values.
    high_dimensional_correction : bool, default = True
        Whether to apply a high-dimensional correction to the estimated coefficients.

    Returns
    -------
    FloatArray
        Rescaled estimated coefficient vector of shape (n_features, 1).
    """
    if not high_dimensional_correction:
        return result.betas

    n_obs, n_features = result.x_validated.shape
    has_intercept = result.intercept_index is not None
    n_params = n_features - int(has_intercept)

    signal_strength = compute_sloe(
        result=result
    )

    pars, _ = solve_state_equation(
        kappa=n_params / n_obs,
        signal_strength=signal_strength,
        alpha=result.alpha,
        corrupted=True,
        intercept=result.theta_hat,
        start=start,
    )

    
    rescaled_betas = result.betas / pars.solution.mu

    if has_intercept:
        rescaled_betas[result.intercept_index, 0] = pars.solution.intercept_estimate

    return rescaled_betas


# TODO: add tests for with and without intercept and with and without starting value
# TODO: add tests the test the full pipeline e.g. fit a model and then call summary
