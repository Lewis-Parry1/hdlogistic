import numpy as np

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
    high_dimensional_correction : bool, optional
        Whether to apply a high-dimensional correction to the estimated coefficients,
        by default True.
    start : FloatArray | None, optional
            Starting values (`mu`, `b`, `sigma`, and optionally `beta_0`) passed to the
            internal state evolution solver (`solve_state_equation`) when
            `high_dimensional_correction=True`. If `None`, defaults to preset values.

    Returns
    -------
    FloatArray
    Rescaled estimated coefficient vector of shape (n_features, 1)
    """
    if high_dimensional_correction:
        number_observations = result.x_validated.shape[0]
        has_intercept = result.theta_hat is not None
        number_parameters = result.x_validated.shape[1] - (1 if has_intercept else 0)

        signal_strength = compute_sloe_estimator(
            linear_predictors=result.linear_predictors,
            y_adjusted=result.y_adjusted,
            leverages=result.leverages,
        )

        kappa = number_parameters / number_observations
        pars, _ = solve_state_equation(
            kappa=kappa,
            signal_strength=signal_strength,
            alpha=result.alpha,
            corrupted=True,
            intercept=result.theta_hat,
            start=start,
        )

        mu_star = pars.solution.mu
        rescaled_betas = result.betas / mu_star

        return np.asarray(rescaled_betas, dtype=np.float64)

    return result.betas


# TODO: add tests for with and without intercept and with and without starting value
# TODO: add tests the test the full pipeline e.g. fit a model and then call summary
