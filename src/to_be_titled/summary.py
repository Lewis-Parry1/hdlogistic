import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

# TODO: Remove this import once refactor is complete and functions are properly exposed
from to_be_titled.estimation import (
    DiaconisYlvisakerLogisticRegressionResult,
)
from to_be_titled.solve_state_equations import solve_state_equation
from to_be_titled.utils import compute_sloe_estimator, compute_weighted_design_and_info


def _compute_leverages(
    x: NDArray[np.float64],
    mus: NDArray[np.float64],
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
    """Compute the diagonal leverage scores (hat values) for a logistic regression
    model.

    Parameters
    ----------
    x : NDArray[np.float64]
        The validated design matrix of shape (n_samples, n_features).
    mus : NDArray[np.float64]
        The fitted probabilities of shape (n_samples, 1).
    epsilon : float, default=1e-8
        Small regularization constant passed to the internal weighting routine.

    Returns
    -------
    NDArray[np.float64]
        The leverage scores (diagonal elements of the hat matrix) for each observation.
    """

    wx, info = compute_weighted_design_and_info(x, mus, epsilon)

    solved_wx_t = solve(info, wx.T, assume_a="pos")

    return np.asarray(np.sum(wx * solved_wx_t.T, axis=1))


def summary(
    result: DiaconisYlvisakerLogisticRegressionResult,
    high_dimensional_correction: bool = True,
) -> NDArray[np.float64]:
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

    Returns
    -------
    NDArray[np.float64]
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
            start=np.array([0.5, 1, 1]),  # Argument required by
            # _solve_state_equation
            gh=None,
            corrupted=True,
        )

        mu_star = pars.solution.mu

        rescaled_betas = result.betas / mu_star

        return np.asarray(rescaled_betas, dtype=np.float64)

    return result.betas
