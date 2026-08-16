from typing import Literal

import numpy as np
from scipy.linalg import solve_triangular

from to_be_titled.types import DYLogisticRegressionResult

def compute_taus(result : DYLogisticRegressionResult):
    """
    ...

    Parameters
    ----------
    result : DYLogisticRegressionResult
        A dataclass which hold the results of the Diaconis-Ylvisaker 
        logistic regression fit obtained from fit_DY_logistic_regression().
    """
    mat_X = (
        result.x_validated if result.intercept_index 
        is None else np.delete(result.x_validated, result.intercept_index, axis = 1)
        ) 

    n, p = mat_X.shape[0], mat_X.shape[1]

    _, mat_R = np.linalg.qr(mat_X)

    mat_R_inv_T = solve_triangular(
        mat_R, 
        np.eye(p),
        trans="T", 
        lower = False
    )

    rss = 1.0 / np.sum(mat_R_inv_T**2, axis = 0)

    return np.sqrt(rss / (n - p + 1.0))
     

def compute_sloe_estimator(result: DYLogisticRegressionResult) -> float:
    """
    Estimate the corrupted signal strength in a model with (sub-)Gaussian covariates.

    The Signal Strength Leave-One-Out Estimator (SLOE) is defined in
    Yadlowsky et al. (2021) when the model is estimated using maximum
    likelihood (i.e., when the shrinkage parameter alpha = 1). The SLOE
    adaptation when estimation is through maximum Diaconis-Ylvisaker prior
    penalized likelihood has been put forward in Sterzinger & Kosmidis (2026).

    In particular, `compute_sloe_estimator` computes an estimate of the
    corrupted signal strength which is the limit: nu^2

    of var(X^T beta(alpha)), where beta(alpha) is the
    maximum Diaconis-Ylvisaker prior penalized likelihood (MDYPL) estimator
    with shrinkage parameter alpha.

    Parameters
    ----------
    result: DYLogisticRegressionResult
        A dataclass which hold the results of the Diaconis-Ylvisaker 
        logistic regression fit obtained from fit_DY_logistic_regression().

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
    logistic_variances = result.mus * (1.0 - result.mus)

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # Ignore warnings for division by zero and invalid operations
        loo_adjusted_predictors = result.linear_predictors - (
            ((result.y_adjusted - result.mus) / logistic_variances)
            * (result.leverages / (1.0 - result.leverages))
        )

    finite_adjusted_predictors = loo_adjusted_predictors[
        np.isfinite(loo_adjusted_predictors)
    ]

    return float(np.std(finite_adjusted_predictors, ddof=1))


def _derive_nu_from_gamma(kappa: float, gamma: float, mu: float, sigma: float) -> float:
    return float(np.sqrt(mu**2 * gamma**2 + kappa * sigma**2))


def _derive_gamma_from_nu(kappa: float, nu: float, sigma: float, mu: float) -> float:
    with np.errstate(invalid="ignore"):
        return float(np.sqrt(nu**2 - kappa * sigma**2) / mu)
