from to_be_titled.inference import compute_sloe_estimator
from to_be_titled.solvers import solve_state_equation
from to_be_titled.types import (
    DiaconisYlvisakerLogisticRegressionResult,
    FloatArray,
)


def summary(
    result: DiaconisYlvisakerLogisticRegressionResult,
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

    signal_strength = compute_sloe_estimator(
        linear_predictors=result.linear_predictors,
        y_adjusted=result.y_adjusted,
        leverages=result.leverages,
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
        rescaled_betas[result.intercept_index, 0] = pars.solution.beta_0

    return rescaled_betas


# TODO: add tests for with and without intercept and with and without starting value
# TODO: add tests the test the full pipeline e.g. fit a model and then call summary
