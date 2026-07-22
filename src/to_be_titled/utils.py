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
        A scalar estimating the corrupted signal strength (nu).

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
