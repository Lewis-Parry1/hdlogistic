import numpy as np 

from typing import Literal
from scipy.special import expit

from to_be_titled.types import FloatArray



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
    2-D design matrix of shape (n_samples, n_features) or 1-D array of shape
    (n_samples,) for single-feature models.
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
