import numpy as np
from scipy.linalg import solve

from to_be_titled.inference import compute_sloe_estimator
from to_be_titled.matrix_operations import compute_weighted_design_and_info
from to_be_titled.solvers import solve_state_equation
from to_be_titled.types import (
    DiaconisYlvisakerLogisticRegressionResult,
    FloatArray,
)


def _compute_leverages(
    x: FloatArray,
    mus: FloatArray,
    epsilon: float = 1e-8,
) -> FloatArray:
    """Compute the diagonal leverage scores (hat values) for a logistic regression
    model.

    Parameters
    ----------
    x : FloatArray
        The validated design matrix of shape (n_samples, n_features).
    mus : FloatArray
        The fitted probabilities of shape (n_samples, 1).
    epsilon : float, default=1e-8
        Small regularization constant passed to the internal weighting routine.

    Returns
    -------
    FloatArray
        The leverage scores (diagonal elements of the hat matrix) for each observation.
    """

    wx, info = compute_weighted_design_and_info(x, mus, epsilon)

    solved_wx_t = solve(info, wx.T, assume_a="pos")

    return np.asarray(np.sum(wx * solved_wx_t.T, axis=1))


def summary(
    result: DiaconisYlvisakerLogisticRegressionResult,
    high_dimensional_correction: bool = True,
    start: FloatArray | None = None,
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
        number_parameters = result.x_validated.shape[1]

        leverages = _compute_leverages(result.x_validated, result.mus)
        signal_strength = compute_sloe_estimator(
            linear_predictors=result.linear_predictors,
            y_adjusted=result.y_adjusted,
            leverages=leverages,
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


# TODO: add tests for with and without intercept
