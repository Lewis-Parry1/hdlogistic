from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class DiaconisYlvisakerLogisticRegressionResult:
    """
    A dataclass to hold the results of the Diaconis-Ylvisaker logistic regression fit.

    Attributes
    ----------
    betas : NDArray[np.float64]
        Estimated coefficient vector of shape (n_features, 1).
    linear_predictors : NDArray[np.float64]
        The fitted linear predictors (eta = X*beta) from the model.
    mus : NDArray[np.float64]
        The fitted probabilities of shape (n_samples, 1).
    y_adjusted : NDArray[np.float64]
        The adjusted or true binary response vector (y).
    x_validated : NDArray[np.float64]
        The validated design matrix of shape (n_samples, n_features).
    alpha : float
        Prior shrinkage hyperparameter used in the model fit, in the interval [0, 1].
    """

    betas: NDArray[np.float64]
    linear_predictors: NDArray[np.float64]
    mus: NDArray[np.float64]
    y_adjusted: NDArray[np.float64]
    x_validated: NDArray[np.float64]
    alpha: float
