import numpy as np
from scipy.linalg import solve
from scipy.special import expit

from to_be_titled.matrix_operations import compute_weighted_design_and_info
from to_be_titled.solvers.registry import register_solver
from to_be_titled.types import FloatArray, LogisticRegressionResult


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


@register_solver("fisher_scoring")
def fit_logistic_regression_fisher_scoring(
    x: FloatArray,
    y: FloatArray,
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
        x : FloatArray
            2-D design matrix of shape (n_samples, n_features).
        y : FloatArray
            2-D response vector of shape (n_samples, 1).
        max_iterations : int, optional
            Maximum number of Fisher scoring iterations (default is 25).
        tolerance : float, optional
            Convergence tolerance threshold for absolute maximum step size
            (default is 1e-6).
        epsilon : float, optional
            Small positive constant passed to the information matrix computation for
            numerical stability (default is 1e-8).

        Returns
        -------
        LogisticRegressionResult
            A dataclass containing the estimated coefficient vector, linear predictors,
            and fitted probabilities.

    Raises
        ------
        TypeError
            If `max_iterations` is not an integer, or if `tolerance` / `epsilon`
            are not real-valued scalars.
        ValueError
            If `max_iterations <= 0`, `tolerance <= 0.0`, or `epsilon <= 0.0`.
        LinAlgError
            If the Fisher information matrix is singular or ill-conditioned and
            cannot be solved during the scoring iteration.
    """
    if isinstance(max_iterations, bool) or not isinstance(
        max_iterations, (int, np.integer)
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"max_iterations must be an integer, got {type(max_iterations).__name__}."
        )
    if max_iterations <= 0:
        raise ValueError(
            f"max_iterations must be a positive integer (> 0), got {max_iterations}."
        )

    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (float, int, np.floating, np.integer)
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"tolerance must be a real-valued number, got {type(tolerance).__name__}."
        )
    if tolerance <= 0.0:
        raise ValueError(
            f"tolerance must be strictly positive (> 0.0), got {tolerance}."
        )
    tolerance = float(tolerance)

    if isinstance(epsilon, bool) or not isinstance(
        epsilon, (float, int, np.floating, np.integer)
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"epsilon must be a real-valued number, got {type(epsilon).__name__}."
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be strictly positive (> 0.0), got {epsilon}.")
    epsilon = float(epsilon)

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
