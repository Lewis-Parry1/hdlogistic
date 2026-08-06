import numpy as np
from scipy.special import expit

from to_be_titled.types import FloatArray, LogisticRegressionResult


def _check_convergence(velocity: FloatArray, tolerance: float) -> bool:
    """Evaluate if the maximum absolute parameter step size falls below tolerance."""
    return bool(np.max(np.abs(velocity)) < tolerance)


def fit_logistic_regression_nesterov_accelerated_gradient_descent(
    x: FloatArray,
    y: FloatArray,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    learning_rate: float = 0.1,
) -> LogisticRegressionResult:
    """Fit a logistic regression model using Nesterov Accelerated Gradient Descent.

    Estimates the regression coefficients by iteratively updating the parameter vector
    using the gradient of the log-likelihood function. Nesterov's method accelerates
    convergence by incorporating a momentum term based on the previous update.
    Convergence is determined by the maximum absolute change in the parameter estimates
    falling below a specified threshold.

    Parameters
    ----------
    x : FloatArray
        2-D design matrix of shape (n_samples, n_features).
    y : FloatArray
        2-D response vector of shape (n_samples, 1).
    max_iterations : int, default = 500
        Maximum number of solver iterations.
    tolerance : float, default = 1e-6
        Convergence tolerance threshold for absolute maximum step size.
    learning_rate : float, default = 0.1
        Base step size (eta) for gradient descent updates.

    Returns
    -------
    LogisticRegressionResult
        A dataclass containing the estimated coefficient vector, linear predictors,
        and fitted probabilities evaluated at the terminal parameter estimates.

    References
    ----------
    .. [1] Nesterov, Y. E. (1983). A method for solving the convex programming problem
       with convergence rate O(1/k^2). *Doklady Akademii Nauk SSSR*, 269(3), 543-547.
       (English translation: *Soviet Mathematics Doklady*, 27(2), 372-376).
    """
    p = x.shape[1]
    n = x.shape[0]

    scaled_learning_rate = learning_rate / n

    betas = np.zeros((p, 1), dtype=np.float64)
    velocity = np.zeros_like(betas)
    t = 1.0

    for _ in range(max_iterations):
        # Solves t_{k+1}^2 - t_{k+1} - t_k^2 = 0 to guarantee O(1/k^2) convergence
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0

        # Calculate momentum blending factor gamma_k = (t_k - 1) / t_{k+1}
        damping_factor = (t - 1.0) / t_next
        t = t_next

        # Project parameters forward using past momentum
        betas_lookahead = betas + damping_factor * velocity

        # Evaluate gradient of negative log-likelihood at look-ahead point
        mus_lookahead = expit(x @ betas_lookahead)
        gradient_lookahead = x.T @ (mus_lookahead - y)

        # Update velocity step and parameters
        velocity = (damping_factor * velocity) - (
            scaled_learning_rate * gradient_lookahead
        )
        betas += velocity

        # Assess convergence bounds
        if _check_convergence(velocity, tolerance):
            break

    # Compute the final linear predictors and fitted probabilities at the terminal
    # parameter estimates.
    etas = x @ betas
    mus = expit(etas)

    return LogisticRegressionResult(
        betas=betas,
        mus=mus,
        linear_predictors=etas,
    )
