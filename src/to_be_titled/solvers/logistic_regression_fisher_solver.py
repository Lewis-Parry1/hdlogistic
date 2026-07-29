import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.special import expit

from to_be_titled.matrix_operations import compute_weighted_design_and_info
from to_be_titled.solvers.solver_types import LogisticRegressionResult


def _compute_fisher_scoring_components(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    betas: NDArray[np.float64],
    epsilon: float,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Compute the Fisher information matrix and the score vector.

    Parameters
    ----------
    x : NDArray[np.float64]
        Design matrix of shape (n_samples, n_features).
    y : NDArray[np.float64]
        Response vector of shape (n_samples, 1).
    betas : NDArray[np.float64]
        Current parameter estimates vector of shape (n_features, 1).
    epsilon : float
        Small positive constant for numerical stability and ridge regularization.

    Returns
    -------
    info : NDArray[np.float64]
        Expected Fisher information matrix of shape (n_features, n_features).
    score : NDArray[np.float64]
        Score function vector (gradient of the log-likelihood) of shape
        (n_features, 1).
    mus : NDArray[np.float64]
        The fitted probabilities of shape (n_samples, 1).
    etas : NDArray[np.float64]
        The fitted linear predictors of shape (n_samples, 1).
    """
    etas = x @ betas
    mus = expit(etas)

    _, info = compute_weighted_design_and_info(x, mus, epsilon)

    score = x.T @ (y - mus)

    return info, score, mus, etas


def fit_logistic_regression_fisher_scoring(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_iterations: int = 25,
    tolerance: float = 1e-6,
    epsilon: float = 1e-8,
) -> LogisticRegressionResult:
    """Fit a logistic regression model using the Fisher scoring method.

    Estimates regression coefficients by iteratively updating the parameter
    vector using the score function and the Fisher information matrix.
    Convergence is determined by the maximum absolute change in the parameter
    estimates falling below a specified threshold.

    Parameters
    ----------
    x : NDArray[np.float64]
        2-D design matrix of shape (n_samples, n_features).
    y : NDArray[np.float64]
        2-D response vector of shape (n_samples, 1).
    max_iterations : int, default=25
        Maximum number of Fisher scoring iterations to perform.
    tolerance : float, default=1e-6
        Convergence tolerance threshold for the absolute maximum step size.
    epsilon : float, default=1e-8
        Small positive constant passed to the information matrix computation.

    Returns
    -------
    LogisticRegressionResult
        A dataclass containing the estimated coefficient vector, linear predictors,
        and fitted probabilities.

    Raises
    ------
    LinAlgError
        If the Fisher information matrix is singular or ill-conditioned and
        cannot be solved during the scoring iteration.
    """
    p = x.shape[1]

    # Initialise the coefficient vector to our initial guess
    betas = np.zeros((p, 1), dtype=np.float64)

    # Initialize outputs
    info, score, mus, etas = _compute_fisher_scoring_components(x, y, betas, epsilon)

    for _ in range(max_iterations):
        info, score, mus, etas = _compute_fisher_scoring_components(
            x, y, betas, epsilon
        )

        step = solve(info, score, assume_a="pos")

        # Assess convergence bounds
        if np.max(np.abs(step)) < tolerance:
            break

        betas += step

    return LogisticRegressionResult(
        betas=betas,
        mus=mus,
        linear_predictors=etas,
    )
