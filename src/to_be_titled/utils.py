from functools import cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, roots_hermite


# TODO: Make this public or properly expose it in the package.
@cache
def _get_hermite_roots_weights(
    n: int = 200,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes and caches the roots and weights of a Hermite polynomial to be
    used in Gauss-Hermite quadrature to approximate an integral.

    Parameters
    ----------
    n : int, optional
        Number of nodes to be used in Gauss Hermite quadrature , by default 200.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Tuple of nodes and corresponding weights to approximate integrals.
    """
    return roots_hermite(n)


def compute_sloe_estimator(
    linear_predictors: NDArray[np.float64],
    y_adjusted: NDArray[np.float64],
    leverages: NDArray[np.float64],
) -> float:
    """
    Estimate the corrupted signal strength in a model with (sub-)Gaussian covariates.

    The Signal Strength Leave-One-Out Estimator (SLOE) is defined in
    Yadlowsky et al. (2021) when the model is estimated using maximum
    likelihood (i.e., when the shrinkage parameter alpha = 1). The SLOE
    adaptation when estimation is through maximum Diaconis-Ylvisaker prior
    penalized likelihood has been put forward in Sterzinger & Kosmidis (2025).

    In particular, `compute_sloe_estimator` computes an estimate of the
    corrupted signal strength which is the limit: nu^2

    of var(X^T beta(alpha)), where beta(alpha) is the
    maximum Diaconis-Ylvisaker prior penalized likelihood (MDYPL) estimator
    with shrinkage parameter alpha.

    Parameters
    ----------
    linear_predictors : NDArray[np.float64]
        The fitted linear predictors (eta = X*beta) from the model.
    y_adjusted : NDArray[np.float64]
        The adjusted or true binary response vector (y).
    leverages : NDArray[np.float64]
        The diagonal elements of the hat matrix (h, leverage values).

    Returns
    -------
    float
        A scalar estimating the corrupted signal strength limit (nu).

    References
    ----------
    .. [1] Sterzinger, P., & Kosmidis, I. (2024). Diaconis-Ylvisaker prior
       penalized likelihood for p/n -> kappa in (0,1) logistic regression.
       arXiv preprint arXiv:2311.07419.
    .. [2] Yadlowsky, S., Yun, T., McLean, C. Y., D'Amour, A. (2021). SLOE: A Faster
       Method for Statistical Inference in High-Dimensional Logistic Regression.
       Advances in Neural Information Processing Systems, 34, 29517–29528.

    """
    predicted_probabilities = expit(linear_predictors)

    logistic_variances = predicted_probabilities * (1.0 - predicted_probabilities)

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # Ignore warnings for division by zero and invalid operations
        loo_adjusted_predictors = linear_predictors - (
            ((y_adjusted - predicted_probabilities) / logistic_variances)
            * (leverages / (1.0 - leverages))
        )

    finite_adjusted_predictors = loo_adjusted_predictors[
        np.isfinite(loo_adjusted_predictors)
    ]

    return float(np.std(finite_adjusted_predictors, ddof=1))


def compute_weighted_design_and_info(
    x: NDArray[np.float64],
    mus: NDArray[np.float64],
    epsilon: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the square-root weighted design matrix and Fisher information matrix.

    Calculates working weights from predicted probabilities, applies
    them element-wise to scale the design matrix, and computes the regularised
    information matrix.

    Parameters
    ----------
    x : NDArray[np.float64]
        Design matrix of shape (n_samples, n_features).
    mus : NDArray[np.float64]
        Predicted mean responses (probabilities) of shape (n_samples, 1) or
        (n_samples,), with values in the interval (0, 1).
    epsilon : float, default=1e-8
        Small positive constant added to the diagonal of the information matrix
        for numerical stability and ridge regularization.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        A tuple containing:
        - wx : Weighted design matrix of shape (n_samples, n_features).
        - info : Regularized Fisher information matrix of shape
        (n_features, n_features).
    """
    p = x.shape[1]

    working_weights = mus * (1.0 - mus)
    wx = np.sqrt(working_weights) * x
    info = wx.T @ wx + np.eye(p) * epsilon

    return wx, info
