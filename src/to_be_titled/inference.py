from typing import cast

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import betaln 

from to_be_titled.types import MDYPLResults, FloatArray

def compute_taus(result: MDYPLResults) -> FloatArray:
    """
    TODO:

    Parameters
    ----------
    result : DYLogisticRegressionResult
        A dataclass which hold the results of the Diaconis-Ylvisaker 
        logistic regression fit obtained from fit_DY_logistic_regression().
    """
    x = cast(FloatArray, result.model.exog)
    n = x.shape[0]
    
    mat_X = (
        x if result.intercept_idx 
        is None else np.delete(x, result.intercept_idx, axis = 1)
        ) 

    p = mat_X.shape[1]

    _, mat_R = np.linalg.qr(mat_X)

    # TODO: Rank defciency check needed 

    mat_R_inv_T = solve_triangular(
        mat_R, 
        np.eye(p),
        trans="T", 
        lower = False
    )

    rss = 1.0 / np.sum(mat_R_inv_T**2, axis = 0)

    return np.sqrt(rss / (n - p + 1.0))
     

def compute_sloe(result: MDYPLResults) -> float:
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
    result: MDYPLResults
        A dataclass which hold the results of the Diaconis-Ylvisaker 
        logistic regression fit.

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
    bernoulli_variances = result.fitted_probs * (1.0 - result.fitted_probs)

    leverages = result.leverages

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # Ignore warnings for division by zero and invalid operations
        sloe_scores = result.linear_predictors - (
            ((result.y_adj - result.fitted_probs) / bernoulli_variances)
            * (leverages / (1.0 - leverages))
        )

    finite_scores = sloe_scores[
        np.isfinite(sloe_scores)
    ]

    return float(np.std(finite_scores, ddof=1))


def _derive_nu_from_gamma(kappa: float, gamma: float, mu: float, sigma: float) -> float:
    return float(np.sqrt(mu**2 * gamma**2 + kappa * sigma**2))


def _derive_gamma_from_nu(kappa: float, nu: float, sigma: float, mu: float) -> float:
    with np.errstate(invalid="ignore"):
        return float(np.sqrt(nu**2 - kappa * sigma**2) / mu)

def _dy_binomial_coeffcient(
        x: FloatArray,
        size: FloatArray | float,
        prob: FloatArray,
        log: bool = False,
) -> FloatArray:
    r"""Generalised binomial probability mass function which allows non-integer x. 

    Parameters
    ----------
    x : FloatArray
        Success rate. 
    size : FloatArray | float
        _description_
    prob : FloatArray | float
    log : bool, optional
        _description_, by default False

    Returns
    -------
    FloatArray
        _description_
    """
    size = np.asarray(size, dtype=float)
    prob = np.asarray(prob, dtype=float)

    successes = x
    failures = size - x 

    # betaln is natual log of absolute value of beta function 
    # allows us to find log(\frac{m!}{s!(m-s)!}) ie. log of binomial coeffcient
    log_db = (
        successes * np.log(prob)
        + failures * np.log(1 - prob)
        - betaln(successes + 1.0, failures + 1.0) 
        - np.log(size + 1.0)
    )

    log_db = np.where(size < successes, -np.inf, log_db)

    if log: 
        return log_db
    return np.exp(log_db)



def _logist_aic(
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
        The -2 log likelihood value. 
    """
    m = freq_weights
    prob_clipped = np.clip(fitted_probs, 1e-8, 1.0 - 1e-8)

    log_likelihood = _dy_binomial_coeffcient(x = m * y_adjusted,
                                             size = m,
                                             prob = prob_clipped,
                                             log = True)

    return float(-2.0 * np.sum(log_likelihood))

   