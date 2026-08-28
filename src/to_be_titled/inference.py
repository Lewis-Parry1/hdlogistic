from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from scipy.special import betaln

if TYPE_CHECKING:
    pass

FloatArray = NDArray[np.float64]


def compute_taus(x: FloatArray, intercept_index: int | None) -> FloatArray:
    """
    TODO:

    Parameters
    ----------
    result : DYLogisticRegressionResult
        A dataclass which hold the results of the Diaconis-Ylvisaker
        logistic regression fit obtained from fit_DY_logistic_regression().
    """

    mat_x = x if intercept_index is None else np.delete(x, intercept_index, axis=1)

    n, p = mat_x.shape[0], mat_x.shape[1]

    _, mat_r = np.linalg.qr(mat_x)

    # TODO: Rank defciency check needed

    mat_r_inv_t = solve_triangular(mat_r, np.eye(p), trans="T", lower=False)

    rss = 1.0 / np.sum(mat_r_inv_t**2, axis=0)

    return np.asarray(np.sqrt(rss / (n - p + 1.0)), dtype=np.float64)


def compute_sloe(
    y_adj: FloatArray,
    linear_predictors: FloatArray,
    fitted_probs: FloatArray,
    leverages: FloatArray,
) -> float:
    """
    Estimate the corrupted signal strength in a model with (sub-)Gaussian covariates.

    The Signal Strength Leave-One-Out Estimator (SLOE) is defined in
    Yadlowsky et al. (2021) when the model is estimated using maximum
    likelihood (i.e., when the shrinkage parameter alpha = 1). The SLOE
    adaptation when estimation is through maximum Diaconis-Ylvisaker prior
    penalized likelihood has been put forward in Sterzinger & Kosmidis (2026).

    In particular, `compute_sloe_estimator` computes an estimate of the
    corrupted signal strength which is the limit: nu^2 of var(X^T beta(alpha)),
    where beta(alpha) is the maximum Diaconis-Ylvisaker prior penalized likelihood
    (MDYPL) estimator with shrinkage parameter alpha.

    Parameters
    ----------
    fitted_probs: FloatArray

    leverages: FloatArray

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
    bernoulli_variances = fitted_probs * (1.0 - fitted_probs)

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # Ignore warnings for division by zero and invalid operations
        sloe_scores = linear_predictors - (
            ((y_adj - fitted_probs) / bernoulli_variances)
            * (leverages / (1.0 - leverages))
        )

    finite_scores = sloe_scores[np.isfinite(sloe_scores)]

    return float(np.std(finite_scores, ddof=1))


def _derive_nu_from_gamma(kappa: float, gamma: float, mu: float, sigma: float) -> float:
    input = mu**2 * gamma**2 + kappa * sigma**2
    if input < 0:
        warnings.warn(
            "Unexpected negative output; nu^2 cannot be negative.", RuntimeWarning
        )
        return float("nan")
    return float(np.sqrt(input))


def _derive_gamma_from_nu(kappa: float, nu: float, sigma: float, mu: float) -> float:
    if abs(mu) < 1e-12:
        warnings.warn(
            "State evolution parameter `mu` is at, near or below zero; "
            "derived signal strength is unreliable.",
            RuntimeWarning,
        )
        return float("nan")

    numerator = nu**2 - kappa * sigma**2
    if numerator < 0.0:
        warnings.warn(
            "Negative value for nu^2 - kappa * sigma ^2, this valuecannot be negative",
            RuntimeWarning,
        )
        return float("nan")

    return float(np.sqrt(numerator) / mu)


def _generalised_binomial_pmf(
    y: FloatArray,
    fw: FloatArray,
    mus: FloatArray,
    log: bool = False,
) -> FloatArray:
    r"""Generalised binomial probability mass function which allows non-integer x.
    This is defiend as, with `s_i` being success rate, `f_i` being failure
    rate, and `fw_i` being the corresponding frequency weight component:

    \mu_i ^ {s_i} * (1 - \mu_i)^{f_i} *
    \frac{\Gamma(fw_i + 1.0)}{\Gamma(s_i + 1.0)\Gamma(f_i + 1.0)}

    Parameters
    ----------
    y : FloatArray
        True response vector used to fit model. Adjusted responsed
        passed in, y_i^* \in [0,1].
    fw : FloatArray | float
        Array of frequency weights used in the fitted model. If vector is not
        vector of 1s then this indicates that certain rows represent more than one
        obersvation.
    mus : FloatArray | float
        Fitted probabilities for each observation.
    log : bool, optional
        If True, the log of the generalised binomial coeffcient is returned
        else the generalised binomail coeffcient is returned, by default False.

    Returns
    -------
    FloatArray
        Generalised binomial PMF corresponding to each row in design matrix.
    """
    size_i = fw

    success_i = y * size_i
    failure_i = size_i - success_i

    log_db = (
        success_i * np.log(mus)
        + failure_i * np.log(1 - mus)
        - betaln(success_i + 1.0, failure_i + 1.0)
        - np.log(size_i + 1.0)
    )

    log_db = np.where(size_i < success_i, -np.inf, log_db)

    if log:
        return log_db
    return np.asarray(np.exp(log_db), dtype=np.float64)


def logist_aic(
    y_adjusted: FloatArray,
    fitted_probs: FloatArray,
    freq_weights: FloatArray,
) -> float:
    """Calculates the -2 log likelihood component of the logistic AIC used
    by MYDPL.

    Parameters
    ----------
    y_adjusted : FloatArray
        True (pseudo-)responses used to fit the model. If the responses are not
        adjusted, this is simply the usual log likelihood for standard logistic
        regression model.
    fitted_probs : FloatArray
        Vector of fitted probabilities found from `MDYPLModel.fit()` and computed
        using rescaled coeffcients if `hd_correction` was True.
    freq_weights : FloatArray
        Frequency weights used to fit model. If no frequency weights were supplied
        this is simply a vector of 1s of shape (n,).

    Returns
    -------
    float
        The -2 * log likelihood value.
    """
    prob_clipped = np.clip(fitted_probs, 1e-8, (1.0 - 1e-8))

    log_likelihood = _generalised_binomial_pmf(
        y_adjusted, freq_weights, prob_clipped, log=True
    )

    return float(-2.0 * np.sum(log_likelihood))
