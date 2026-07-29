from typing import Any

import numpy as np
from scipy.linalg import solve
from scipy.special import expit

from to_be_titled.matrix_operations import compute_weighted_design_and_info
from to_be_titled.types import FloatArray, LogisticRegressionResult


def _validate_config(config: dict[str, Any]) -> None:
    """Validate that convergence parameters in the config are floats."""
    for key in ("max_iterations", "tolerance", "epsilon"):
        if key in config and not isinstance(config[key], (float, int)):
            raise TypeError(
                f"Config parameter '{key}' must be a float, got {
                    type(config[key]).__name__
                }"
            )


def _compute_fisher_scoring_components(
    x: FloatArray,
    y: FloatArray,
    betas: FloatArray,
    epsilon: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute the Fisher information matrix and the score vector.

    Parameters
    ----------
    x : FloatArray
        Design matrix of shape (n_samples, n_features).
    y : FloatArray
        Response vector of shape (n_samples, 1).
    betas : FloatArray
        Current parameter estimates vector of shape (n_features, 1).
    epsilon : float
        Small positive constant for numerical stability and ridge regularization.

    Returns
    -------
    info : FloatArray
        Expected Fisher information matrix of shape (n_features, n_features).
    score : FloatArray
        Score function vector (gradient of the log-likelihood) of shape
        (n_features, 1).
    mus : FloatArray
        The fitted probabilities of shape (n_samples, 1).
    etas : FloatArray
        The fitted linear predictors of shape (n_samples, 1).
    """
    etas = x @ betas
    mus = expit(etas)

    _, info = compute_weighted_design_and_info(x, mus, epsilon)

    score = x.T @ (y - mus)

    return info, score, mus, etas


def fit_logistic_regression_fisher_scoring(
    x: FloatArray, y: FloatArray, config: dict[str, Any]
) -> LogisticRegressionResult:
    """Fit a logistic regression model using the Fisher scoring method.

    Estimates regression coefficients by iteratively updating the parameter
    vector using the score function and the Fisher information matrix.
    Convergence is determined by the maximum absolute change in the parameter
    estimates falling below a specified threshold.

    Parameters
    ----------
    x : FloatArray
        2-D design matrix of shape (n_samples, n_features).
    y : FloatArray
        2-D response vector of shape (n_samples, 1).
    config : dict[str, Any]
        Configuration options dictionary. Supported keys:
        - `"max_iterations"` (int, default=25): Maximum number of Fisher scoring
        iterations.
        - `"tolerance"` (float, default=1e-6): Convergence tolerance threshold for
        absolute maximum step size.
        - `"epsilon"` (float, default=1e-8): Small positive constant passed to the
        information matrix computation.

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

    config = config or {}

    _validate_config(config=config)

    max_iterations = config.get("max_iterations", 25)
    tolerance = config.get("tolerance", 1e-6)
    epsilon = config.get("epsilon", 1e-8)

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
