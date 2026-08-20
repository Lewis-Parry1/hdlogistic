import numpy as np

from to_be_titled.types import FloatArray


def _has_constant_col(x: FloatArray) -> bool:
    """Returns True if a constant column is detected.

    Parameters
    ----------
    x : FloatArray
        Design matrix of shape (`n`, `p`) where `n` is number of observations and
        `p` is number of features.

    Returns
    -------
    bool
        _description_
    """
    return bool(np.any(np.all(x == x[0, :], axis=0)))


def _get_intercept_idx(x: FloatArray) -> int | None:
    """Returns integer representing index of intercept in design matrix
    using (zero-indexing).

    Parameters
    ----------
    x : FloatArray
        Design matrix of shape (`n`, `p`) where `n` is number of observations and
        `p` is number of features.

    Returns
    -------
    int | None
        Integer index of constant/intercept column, if no constant column is detected
        returns None.
    """
    is_const = np.all(x == x[0, :], axis = 0)
    idx = np.flatnonzero(is_const)
    return int(idx[0]) if idx.size else None


def _adjust_response(y: FloatArray, alpha: float) -> FloatArray:
    """Compute the adjusted response vector under a Diaconis-Ylvisaker prior.

    Transforms the empirical binary responses into pseudo-probabilities shifted
    toward the prior distribution. This specific formulation assumes a zero prior
    mode, which evaluates the sigmoid function to 0.5.

    Parameters
    ----------
    y : FloatArray
        Original binary response vector of shape (n_samples,) or (n_samples, 1).
    alpha : float
        Shrinkage parameter in [0, 1]. Lower values enforce stronger prior
        regularization, pulling the pseudo-responses toward 0.5 (which shrinks
        coefficient estimates toward the prior mode of 0). Setting alpha = 1.0
        recovers standard unpenalized maximum likelihood estimation.

    Returns
    -------
    FloatArray
        Adjusted response vector of the same shape as y, with values continuous
        on the interval [0, 1].
    """
    return alpha * y + (1 - alpha) / 2
