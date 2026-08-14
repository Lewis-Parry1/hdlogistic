from typing import Literal

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


def predict(
    x: FloatArray,
    betas: FloatArray,
    *,
    offset: FloatArray | None = None,
    type: Literal["response", "link"] = "response",
) -> FloatArray:
    """Compute predictions for new observations given a design matrix, coefficient
    vector, and optional offset.

    Parameters
    ----------
    x : FloatArray
        1-D or 2-D design matrix of shape (n_samples, n_features) or (n_features,).
    betas : FloatArray
        2-D column vector of regression coefficients of shape (n_features, 1)
        or 1-D array of shape (n_features,).
    offset : FloatArray | None, default = None
        1-D or 2-D array of a priori known components to be included in the
        linear predictor. Must match the number of samples in `x`[cite: 5].
    type : Literal["response", "link"], default = "response"
        Type of prediction to compute[cite: 5]:
        - `"response"`: Fitted probabilities in [0.0, 1.0] via inverse logit[cite: 5].
        - `"link"`: Linear predictors (eta = x @ betas + offset)[cite: 5].

    Returns
    -------
    FloatArray
        2-D column vector of predictions of shape (n_samples, 1)[cite: 5].

    Raises
    ------
    ValueError
        If `type` is not one of `{"response", "link"}`[cite: 5].
        If `x`, `betas`, or `offset` fail structural and dimensional checks.
    Examples
    --------
    >>> result = fit_diaconis_ylvisaker_logistic_regression(
    ...     X_train, y_train, intercept_index=0
    ... )
    >>> # Predict using high-dimensionally corrected coefficients:
    >>> corrected_betas = summary(result, high_dimensional_correction=True)
    >>> y_pred = predict(X_test, corrected_betas)
    >>>
    >>> # Predict using raw, uncorrected Diaconis-Ylvisaker estimates:
    >>> y_pred_raw = predict(X_test, result.betas)
    """
    if type not in {"response", "link"}:
        raise ValueError(
            f"Invalid prediction type '{type}'. Expected 'response' or 'link'."
        )

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    elif x_arr.ndim != 2:
        raise ValueError("x must be a 2-D design matrix.")

    betas_arr = np.asarray(betas, dtype=np.float64)
    if betas_arr.ndim == 1:
        betas_arr = betas_arr.reshape(-1, 1)
    elif betas_arr.ndim != 2 or betas_arr.shape[1] != 1:
        raise ValueError("betas must be a 1-D array or 2-D column vector.")

    n_samples, n_features = x_arr.shape

    if betas_arr.shape[0] != n_features:
        raise ValueError(
            f"Number of coefficients ({betas_arr.shape[0]}) does not match "
            f"number of features in x ({n_features})."
        )

    eta = x_arr @ betas_arr

    if offset is not None:
        offset_arr = np.asarray(offset, dtype=np.float64)
        if offset_arr.ndim == 1:
            offset_arr = offset_arr.reshape(-1, 1)
        elif offset_arr.ndim != 2 or offset_arr.shape[1] != 1:
            raise ValueError("offset must be a 1-D array or 2-D column vector.")

        if offset_arr.shape[0] != n_samples:
            raise ValueError(
                f"Offset length ({offset_arr.shape[0]}) does not match "
                f"number of samples in x ({n_samples})."
            )

        eta += offset_arr

    if type == "link":
        return eta

    return np.asarray(expit(eta), dtype=np.float64)


def _derive_nu_from_gamma(kappa: float, gamma: float, mu: float, sigma: float) -> float:
    return float(np.sqrt(mu**2 * gamma**2 + kappa * sigma**2))


def _derive_gamma_from_nu(kappa: float, nu: float, sigma: float, mu: float) -> float:
    with np.errstate(invalid="ignore"):
        return float(np.sqrt(nu**2 - kappa * sigma**2) / mu)
