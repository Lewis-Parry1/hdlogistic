from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from scipy.special import betaln, xlogy

FloatArray = NDArray[np.float64]


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

    Returns
    -------
    FloatArray
        Array of conditional standard deviations \tau_j for each j=1,...p.
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
    y_adj : FloatArray
        Adjusted response vector of shape `(n,)` used to fit the model,
        `y_adj = alpha * y_raw + (1 - alpha) / 2`. The SLOE estimator is
        always evaluated on this scale, consistent with the penalised
        likelihood the model was fit on.
    linear_predictors : FloatArray
        Linear predictors `X @ beta + offset` of shape `(n,)` from the
        fitted (non-rescaled) model.
    fitted_probs : FloatArray
        Fitted probabilities of shape `(n,)` from the fitted
        (non-rescaled) model.
    leverages : FloatArray
        Diagonal entries of the hat matrix, of shape `(n,)`.

    Returns
    -------
    float
        A scalar estimating the square root of the corrupted signal
        strength.

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
    r"""Derive `nu`, the square root of the corrupted signal strength, from
    `gamma`, the square root of the true signal strength, and the state
    evolution parameters.

    Computes `nu = sqrt(mu^2 * gamma^2 + kappa * sigma^2)`, the inverse
    relationship to `derive_gamma_from_nu`, following the state evolution
    equations in Sterzinger & Kosmidis (2024).

    Parameters
    ----------
    kappa : float
        Aspect ratio `p / nobs_eff` (ratio of predictors to effective
        sample size).
    gamma : float
        The square root of the true signal strength, i.e. the square
        root of the limit of `var(X @ beta)`.
    mu : float
        State evolution parameter `mu` from `solve_state_equation`.
    sigma : float
        State evolution parameter `sigma` from `solve_state_equation`.

    Returns
    -------
    float
        The derived `nu`, the square root of the corrupted signal
        strength, or `nan` (with a `RuntimeWarning`) if the computation
        would require the square root of a negative number.
    """
    input = mu**2 * gamma**2 + kappa * sigma**2
    if input < 0:
        warnings.warn(
            "Unexpected negative output; nu^2 cannot be negative.", RuntimeWarning
        )
        return float("nan")
    return float(np.sqrt(input))


def derive_gamma_from_nu(kappa: float, nu: float, sigma: float, mu: float) -> float:
    r"""Derive `gamma`, the square root of the true signal strength, from
    `nu`, the square root of the corrupted signal strength (e.g. from
    `compute_sloe`), and the state evolution parameters.

    Computes `gamma = sqrt(nu^2 - kappa * sigma^2) / mu`. Used to
    obtain `signal_strength = gamma**2` in the high-dimensional summary
    diagnostics.

    Parameters
    ----------
    kappa : float
        Aspect ratio `p / nobs_eff` (ratio of predictors to effective
        sample size).
    nu : float
        Square root of the corrupted signal strength estimate. To
        obtain this quantity, see `compute_sloe()`.
    sigma : float
        State evolution parameter `sigma` from `solve_state_equation`.
    mu : float
        State evolution parameter `mu` from `solve_state_equation`.

    Returns
    -------
    float
        The derived `gamma`, the square root of the true signal strength,
        or `nan` (with a `RuntimeWarning`) if `mu` is at or near zero, or
        if the computation would require the square root of a negative
        number.
    """
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
    binomial probability mass function which allows non-integer y.
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


def compute_aic(
    log_likelihood: float,
    rank: int,
) -> float:
    """Calculates the AIC for a logistic regression model given a
    precomputed total log-likelihood. The AIC is defined as
    -2 * log likelihood + 2 * rank.

    Parameters
    ----------
    log_likelihood : float
        Total log-likelihood of the fitted model. Run `compute_likelihood`
        (with `log=True`) beforehand to obtain this value, e.g.
        `compute_likelihood(y, freq_weights, fitted_probs, log=True)`.
        Passed in directly rather than recomputed here, since callers
        typically already need the log-likelihood separately (e.g. as
        `llf`) and recomputing it a second time inside this function
        would duplicate that work.
    rank : int
        The rank of the design matrix used to fit the model. This is used to
        compute the AIC penalty term.

    Returns
    -------
    float
        The -2 * log likelihood value + 2 * rank,
        which is the AIC for the fitted model.
    """
    aic = -2.0 * log_likelihood + 2.0 * rank

    return float(aic)


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
