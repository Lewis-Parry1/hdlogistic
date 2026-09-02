from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import betaln, xlogy

from to_be_titled.types import FloatArray


def compute_taus(x: FloatArray, intercept_index: int | None) -> FloatArray:
    r"""
    Computes \tau_j which is \text{var}(x_{,j}|\textbf x_{,-j}). In other
    words this function returns an array of \tau_j for each covariate j, where
    \tau_j is the conditional variance of covariate j conditioned on all remianing
    covariates.

    Parameters
    ----------
    x: FloatArray
        Full rank design matrix used to fit MDYPL model
        (with intercept if included).
    intercept_index: int | None,
        Index of intercept column of design matrix, if an intercept is included.
        By default, None.
    """

    mat_x = x if intercept_index is None else np.delete(x, intercept_index, axis=1)

    n, p = mat_x.shape[0], mat_x.shape[1]

    _, mat_r = np.linalg.qr(mat_x)

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


def derive_nu_from_gamma(kappa: float, gamma: float, mu: float, sigma: float) -> float:
    input = mu**2 * gamma**2 + kappa * sigma**2
    if input < 0:
        warnings.warn(
            "Unexpected negative output; nu^2 cannot be negative.", RuntimeWarning
        )
        return float("nan")
    return float(np.sqrt(input))


def derive_gamma_from_nu(kappa: float, nu: float, sigma: float, mu: float) -> float:
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
            "Negative value for nu^2 - kappa * sigma ^2, this value cannot be negative",
            RuntimeWarning,
        )
        return float("nan")

    return float(np.sqrt(numerator) / mu)


def compute_likelihood(
    y: FloatArray,
    fw: FloatArray,
    mus: FloatArray,
    log: bool = False,
    eps: float = 1e-15,
) -> float:
    r"""Compute the total log likelihood using the generalised
    binomial probability mass function which allows non-integer x.
    This is defined as, with `s_i` being success rate,
    `f_i` being failure rate, and `fw_i` being the corresponding frequency
    weight component:

    Binomial PMF for i = \mu_i ^ {s_i} * (1 - \mu_i)^{f_i} *
    \frac{\Gamma(fw_i + 1.0)}{\Gamma(s_i + 1.0)\Gamma(f_i + 1.0)}

    Returns the sum over all n * fw observations, which is the log-likelihood.

    Parameters
    ----------
    y : FloatArray
        True response vector used to fit model. Adjusted responses
        passed in, y_i \in [0,1]. Binary responses also supported.
    fw : FloatArray | float
        Array of frequency weights used in the fitted model. If vector is not
        vector of 1s then this indicates that certain rows represent more than one
        observation.
    mus : FloatArray | float
        Fitted probabilities for each observation.
    log : bool, optional
        If True, the log of the generalised binomial coefficient is returned
        else the generalised binomial coefficient is returned, by default False.
    eps: float, optional
        Small value to avoid log(0) issues, by default 1e-15.
    Returns
    -------
    float
        Total log-likelihood.
    """
    size_i = fw

    success_i = y * size_i
    failure_i = size_i - success_i

    mus_clipped = np.clip(mus, eps, 1.0 - eps)

    # Array of individual log-likelihoods
    log_db = (
        success_i * np.log(mus_clipped)
        + failure_i * np.log(1 - mus_clipped)
        - betaln(success_i + 1.0, failure_i + 1.0)
        - np.log(size_i + 1.0)
    )

    log_db = np.where(size_i < success_i, -np.inf, log_db)
    total_log_like = np.sum(log_db)  # Get sum of log - likelihood

    if log:
        return total_log_like

    return np.exp(total_log_like)


def logistic_aic(
    y: FloatArray,
    fitted_probs: FloatArray,
    freq_weights: FloatArray,
    rank: int,
    eps: float = 1e-15,
) -> float:
    """Calculates the AIC correctly for a logistic regression model, even when
    the responses are [0,1] and not just binary. The AIC is defined as
    -2 * log likelihood + 2 * rank, where the log likelihood is the sum of
    the log of the generalised binomial PMF for each observation.

    Parameters
    ----------
    y : FloatArray
        True (pseudo-)responses used to fit the model. If the responses are not
        adjusted, this is simply the usual log likelihood for standard logistic
        regression model.
    fitted_probs : FloatArray
        Vector of fitted probabilities found from `MDYPLModel.fit()` and computed
        using rescaled coefficients if `hd_correction` was True.
    freq_weights : FloatArray
        Frequency weights used to fit model, which are the per observation trial
        counts. If no frequency weights were supplied this is simply a vector of
        1s of shape (n,).
    rank : int
        The rank of the design matrix used to fit the model. This is used to
        compute the AIC penalty term.
    eps : float, optional
        Small value to avoid log(0) issues, by default 1e-15.

    Returns
    -------
    float
        The -2 * log likelihood value + 2 * rank,
        which is the AIC for the fitted model.
    """
    log_likelihood = compute_likelihood(
        y, freq_weights, fitted_probs, log=True, eps=eps
    )
    aic = -2.0 * log_likelihood + 2.0 * rank

    return float(aic)


def logistic_bic(
    y: np.ndarray,
    fitted_probs: np.ndarray,
    freq_weights: np.ndarray,
    rank: int,
    eps: float = 1e-15,
) -> float:
    r"""
    Computes the Bayesian Information Criterion (BIC)
    for a logistic regression model.

    The BIC is defined as:
        BIC = -2 * log likelihood + log(n) * rank,
    where the log likelihood is the sum of the log of
    the generalised binomial PMF for each observation.

    Parameters
    ----------
    y : np.ndarray
        True response vector used to fit model. Adjusted responsed
        passed in, are in [0,1]. Binary responses also supported.
    fitted_probs : np.ndarray
        Vector of fitted probabilities found from `MDYPLModel.fit()` and computed
        using rescaled coefficients if `hd_correction` was True.
    freq_weights : np.ndarray
        Frequency weights used to fit model, which are the per observation trial
        counts. If no frequency weights were supplied this is simply a vector of
        1s of shape (n,).
    rank : int
        The rank of the design matrix used to fit the model. This is used to
        compute the BIC penalty term.
    eps : float, optional
        Small value to avoid log(0) issues, by default 1e-15.
    """
    log_likelihood = compute_likelihood(
        y, freq_weights, fitted_probs, log=True, eps=eps
    )
    # N is usually the sum of frequency weights (total trials)
    nobs = np.sum(freq_weights)

    bic = -2.0 * log_likelihood + rank * np.log(nobs)

    return float(bic)


def compute_deviance(
    y: FloatArray,
    fitted_probs: FloatArray,
    freq_weights: FloatArray,
    eps: float = 1e-15,
) -> float:
    r"""
    Computes the deviance for a logistic regression model.
    y can be either the raw responses or the adjusted responses.
    The general Binomial deviance formula is,

    D = 2 * \sum w_i *[y_i * \ln (y_i/mu_i) + (1 - y_i) *
    \ln ((1 - y_i)/(1 - mu_i))]
    = 2 * \sum w_i * (LogLik_sat - LogLik_fit)

    where y_i is the observed response, mu_i is the fitted probability, and
    w_i are the frequency weights for each observation. If y is binary
    then this is the usual deviance for a logistic regression model.

    Adds a small epsilon to fitted probabilities to avoid log(0) issues.

    Parameters
    ----------
    y : FloatArray
        True response vector used to fit model. Adjusted responsed
        passed in, y_i \in [0,1]. Binary responses also supported.
    fitted_probs : FloatArray
        Vector of fitted probabilities found from `MDYPLModel.fit()` and computed
        using rescaled coefficient if `hd_correction` was True.
    freq_weights : FloatArray
        Frequency weights used to fit model, which are the per observation trial
        counts. If no frequency weights were supplied this is simply a vector of
        1s of shape (n,).
    eps : float, optional
        Small value to avoid log(0) issues, by default 1e-15.

    Returns
    -------
    float
        The deviance for the fitted model.
    """
    mu_clipped = np.clip(fitted_probs, eps, 1.0 - eps)

    # Use xlogy to compute xlog(y) as it handles log(0)
    saturated_loglike = xlogy(y, y) + xlogy(1 - y, 1 - y)
    fit_loglike = xlogy(y, mu_clipped) + xlogy(1 - y, 1 - mu_clipped)

    deviance = 2.0 * np.sum(freq_weights * (saturated_loglike - fit_loglike))

    return float(deviance)


def compute_deviance_residuals(
    y: FloatArray,
    fitted_probs: FloatArray,
    freq_weights: FloatArray,
    eps: float = 1e-15,
) -> FloatArray:
    r"""
    Computes the deviance residuals for a logistic regression model.
    The deviance residual is defined as the contribution of each
    observation to the total deviance. It is calculated as

    d_i = sign(y_i - mu_i) * sqrt(2 * w_i * [y_i
    * log(y_i/mu_i) + (1 - y_i) * log((1 - y_i)/(1 - mu_i))])
    = sign(y_i - mu_i) * sqrt(2 * w_i * (LogLik_sat_i - LogLik_fit_i))

    Parameters
    ----------
    y : FloatArray
        True response vector used to fit model. Adjusted responsed
        passed in, are in [0,1]. Binary responses also supported.
    fitted_probs : FloatArray
        Vector of fitted probabilities found from `MDYPLModel.fit()` and computed
        using rescaled coefficients if `hd_correction` was True.
    freq_weights : FloatArray
        Frequency weights used to fit model, which are the per observation trial
        counts. If no frequency weights were supplied this is simply a vector of
        1s of shape (n,).
    eps : float, optional
        Small value to avoid log(0) issues, by default 1e-15.

    Returns
    -------
    FloatArray
        Deviance residuals for each observation in the fitted model.

    """
    mu_clipped = np.clip(fitted_probs, eps, 1.0 - eps)

    sat_loglike_i = xlogy(y, y) + xlogy(1 - y, 1 - y)
    fit_loglike_i = xlogy(y, mu_clipped) + xlogy(1 - y, 1 - mu_clipped)

    squared_deviance_residual = 2.0 * freq_weights * (sat_loglike_i - fit_loglike_i)

    abs_deviance_residual = np.sqrt(np.maximum(squared_deviance_residual, 0.0))

    sign_i = np.sign(y - mu_clipped)

    return sign_i * abs_deviance_residual


def compute_pearson_residuals(
    y: np.ndarray,
    fitted_probs: np.ndarray,
    freq_weights: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Computes Pearson residuals:
    r_i = (y - mu) / sqrt(mu * (1 - mu) / w)
    """
    mu_clipped = np.clip(fitted_probs, eps, 1.0 - eps)

    v_i = (mu_clipped * (1.0 - mu_clipped)) / freq_weights
    residuals_raw = y - mu_clipped

    return (residuals_raw) / np.sqrt(np.maximum(v_i, eps))

# TODO: Lewis, when you add print summary can you fill this out
# Just copy confint.mdyplFit in brglm2, use our rescaled (if hd=True)
# Or not rescaled coeffcients (hd = False). 
def get_confidence_interval():
    pass
