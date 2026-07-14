import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.special import expit


def _adjust_response(y: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Compute the adjusted response vector under a Diaconis-Ylvisaker prior.

    Transforms the empirical binary responses into pseudo-probabilities shifted
    toward the prior distribution. This specific formulation assumes a zero prior
    mode, which evaluates the inverse link function to 0.5.

    Parameters
    ----------
    y : NDArray[np.float64]
        Original binary response vector of shape (n_samples,) or (n_samples, 1).
    alpha : float
        Prior shrinkage hyperparameter in [0, 1]. Lower values enforce stronger
        shrinkage toward the prior weight of 0.5; alpha = 1.0 recovers the
        original response vector.

    Returns
    -------
    NDArray[np.float64]
        Adjusted response vector of the same shape as y, with values continuous
        on the interval [0, 1].
    """
    return alpha * y + (1 - alpha) / 2


def _ensure_design_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert input to a 2-D float64 design matrix and validate dimensions.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input feature array-like structure.

    Returns
    -------
    NDArray[np.float64]
        Validated 2-D design matrix of shape (n_samples, n_features).

    Raises
    ------
    ValueError
        If `x` cannot be reshaped into or validated as a 2-D matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError("x must be a 2-D design matrix")
    return x


def _ensure_column_vector(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert input to a 2-D float64 column vector and validate dimensions.

    Parameters
    ----------
    y : NDArray[np.float64]
        Input response array-like structure.

    Returns
    -------
    NDArray[np.float64]
        Validated 2-D column vector of shape (n_samples, 1).

    Raises
    ------
    ValueError
        If `y` cannot be represented as a 1-D vector or a 2-D column vector.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim == 2 and y.shape[1] != 1:
        raise ValueError("y must be a 1-D response vector or a 2-D column vector")
    elif y.ndim > 2:
        raise ValueError("y must be a 1-D response vector or a 2-D column vector")
    return y


def _compute_fisher_scoring_components(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    betas: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the Fisher information matrix and the score vector.

    Parameters
    ----------
    x : NDArray[np.float64]
        Design matrix of shape (n_samples, n_features).
    y : NDArray[np.float64]
        Response vector of shape (n_samples, 1).
    betas : NDArray[np.float64]
        Current parameter estimates vector of shape (n_features, 1).

    Returns
    -------
    info : NDArray[np.float64]
        Expected Fisher information matrix of shape (n_features, n_features).
    score : NDArray[np.float64]
        Score function vector (gradient of the log-likelihood) of shape
        (n_features, 1).

    Raises
    ------
    ValueError
        If matrix dimensions between `x`, `y`, and `betas` are incompatible for
        matrix multiplication.
    """
    p = x.shape[1]  # Number of features in the design matrix

    #TODO: Consider making a parameter or from a global variable/config as it is a 
    # magic number
    epsilon = 1e-8  # Small ridge for numerical stability 


    etas = x @ betas
    mus = expit(etas)

    working_weights = mus * (1.0 - mus)
    wx = np.sqrt(working_weights) * x

    info = wx.T @ wx + np.eye(p) * epsilon  # Add small ridge for numerical stability

    score = x.T @ (y - mus)

    return info, score


def _fit_logistic_regression(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_iterations: int = 25,
    tolerance: float = 1e-6,
) -> NDArray[np.float64]:
    """Fit a logistic regression model using the Fisher scoring method.

    Estimates regression coefficients by iteratively updating the parameter
    vector using the score function and the Fisher information matrix.
    Convergence is determined by the maximum absolute change in the parameter
    estimates falling below a specified threshold.

    Parameters
    ----------
    x : NDArray[np.float64]
        Validated 2-D design matrix of shape (n_samples, n_features).
    y : NDArray[np.float64]
        Validated 2-D response vector of shape (n_samples, 1).
    max_iterations : int, default=25
        Maximum number of Fisher scoring iterations to perform.
    tolerance : float, default=1e-6
        Convergence tolerance threshold for the absolute maximum step size.

    Returns
    -------
    NDArray[np.float64]
        Estimated coefficient vector of shape (n_features, 1).

    Raises
    ------
    LinAlgError
        If the Fisher information matrix is singular or ill-conditioned and
        cannot be solved during the scoring iteration.
    """
    p = x.shape[1]  # Number of features in the design matrix

    # Initialise the coefficient vector to our initial guess
    betas = np.zeros((p, 1), dtype=np.float64)

    for _ in range(max_iterations):
        info, score = _compute_fisher_scoring_components(x, y, betas)

        step = solve(info, score)

        # Assess convergence bounds
        if np.max(np.abs(step)) < tolerance:
            break

        betas += step

    return betas


def fit_diaconis_ylvisaker_logistic_regression(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float = 0.5,
    max_iterations: int = 25,
    tolerance: float = 1e-6,
) -> NDArray[np.float64]:
    """Fit a logistic regression model using a Diaconis-Ylvisaker prior.

    Estimates regression coefficients using a Fisher scoring method. Due to the
    properties of the Diaconis-Ylvisaker prior, simplifies to standard maximisation of
    the log-likelihood function, on an adjusted response vector.

    Parameters
    ----------
    x : NDArray[np.float64]
        Design matrix of shape (n_samples, n_features).
    y : NDArray[np.float64]
        Binary response vector of shape (n_samples,) or (n_samples, 1).
    alpha : float, default=0.5
        Prior shrinkage hyperparameter in [0, 1]. Controls the variance of
        the prior distribution. As alpha approaches 0, estimates shrink toward
        the prior mode, as alpha approaches 1, the maximum likelihood
        estimate is recovered.
    max_iterations : int, default=25
        Maximum number of Fisher scoring iterations to perform.
    tolerance : float, default=1e-6
        Convergence tolerance threshold for the absolute maximum step size.

    Returns
    -------
    NDArray[np.float64]
        Penalized estimated coefficient vector of shape (n_features, 1).

    Raises
    ------
    ValueError
        If `alpha` is not in the closed interval [0.0, 1.0].
        If `x` or `y` fail structural checks during validation.
    LinAlgError
        If the Fisher information matrix is singular during solver operations.

    References
    ----------
    .. [1] Sterzinger, P., & Kosmidis, I. (2024). Diaconis-Ylvisaker prior
           penalized likelihood for p/n -> kappa in (0,1) logistic regression.
           https://arxiv.org/abs/2311.07419
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    x_validated = _ensure_design_matrix(x)
    y_validated = _ensure_column_vector(y)

    y_adjusted = _adjust_response(y_validated, alpha=alpha)

    return _fit_logistic_regression(x_validated, y_adjusted, max_iterations, tolerance)
