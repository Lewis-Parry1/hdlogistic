import numpy as np

from to_be_titled.types import FloatArray


def compute_weighted_design_and_info(
    x: FloatArray,
    mus: FloatArray,
    epsilon: float = 1e-8,
) -> tuple[FloatArray, FloatArray]:
    """Compute the square-root weighted design matrix and Fisher information matrix.

    Calculates working weights from predicted probabilities, applies
    them element-wise to scale the design matrix, and computes the regularised
    information matrix.

    Parameters
    ----------
    x : FloatArray
        Design matrix of shape (n_samples, n_features).
    mus : FloatArray
        Predicted mean responses (probabilities) of shape (n_samples, 1) or
        (n_samples,), with values in the interval (0, 1).
    epsilon : float, default=1e-8
        Small positive constant added to the diagonal of the information matrix
        for numerical stability and ridge regularization.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        A tuple containing:
        - wx : Weighted design matrix of shape (n_samples, n_features).
        - info : Regularized Fisher information matrix of shape
        (n_features, n_features).
    """
    p = x.shape[1]

    working_weights = mus * (1.0 - mus)
    wx = np.sqrt(working_weights) * x
    info = wx.T @ wx + np.eye(p) * epsilon

    return wx, info
