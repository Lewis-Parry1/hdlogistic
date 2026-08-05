from typing import Any

import numpy as np
from scipy.special import expit

from to_be_titled.types import FloatArray, LogisticRegressionResult


def fit_logistic_regression_nesterov_accelerated_gradient_descent(
    x: FloatArray, y: FloatArray, config: dict[str, Any]
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
    config : dict[str, Any]
        Configuration options dictionary. Supported keys:
        - `"max_iterations"` (int, default=500): Maximum number of solver iterations.
        - `"tolerance"` (float, default=1e-6): Convergence tolerance threshold for
          absolute maximum step size.
        - `"learning_rate"` (float, default=0.1): Base step size (eta) for gradient
          descent updates.

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

    config = config or {}

    max_iterations = config.get("max_iterations", 500)
    tolerance = config.get("tolerance", 1e-6)
    learning_rate = config.get("learning_rate", 0.1)

    scaled_learning_rate = learning_rate / x.shape[0]

    p = x.shape[1]

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

            # Compute predictions at the look-ahead position
            etas_lookahead = x @ betas_lookahead
            mus_lookahead = expit(etas_lookahead)

            # Evaluate slope of negative log-likelihood at look-ahead point
            gradient_lookahead = x.T @ (mus_lookahead - y)

            # 5. Velocity Update: Blend damped past momentum with future slope correction
            velocity = damping_factor * velocity - scaled_learning_rate * gradient_lookahead

            # 6. Parameter Update: Apply final displacement vector to current parameters
            betas = betas + velocity

            # Assess convergence bounds
            if np.max(np.abs(velocity)) < tolerance:
                break

    # Compute the final linear predictors and fitted probabilities at the terminal parameter estimates.
    etas = x @ betas
    mus = expit(etas)

    return LogisticRegressionResult(
        betas=betas,
        mus=mus,
        linear_predictors=etas,
    )
