import numpy as np
from scipy.special import expit

from to_be_titled.types import FloatArray


def compute_sloe_estimator(
    linear_predictors: FloatArray,
    y_adjusted: FloatArray,
    leverages: FloatArray,
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
    linear_predictors : FloatArray
        The fitted linear predictors (eta = X*beta) from the model.
    y_adjusted : FloatArray
        The adjusted or true binary response vector (y).
    leverages : FloatArray
        The diagonal elements of the hat matrix (h, leverage values).

    Returns
    -------
    float
        A scalar estimating the corrupted signal strength (nu).

    References
    ----------
    .. [1] Sterzinger, P., & Kosmidis, I. (2026). Diaconis-Ylvisaker prior
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


def _derive_gamma_from_nu(kappa: float, nu: float, mu: float, sigma: float) -> float:
    return float(np.sqrt(nu**2 - kappa * sigma**2) / mu)


def _derive_nu_from_gamma(kappa: float, gamma: float, mu: float, sigma: float) -> float:
    return float(np.sqrt(mu**2 * gamma**2 + kappa * sigma**2))
