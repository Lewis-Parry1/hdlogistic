from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


@dataclass
class DiaconisYlvisakerLogisticRegressionResult:
    """
    A dataclass to hold the results of the Diaconis-Ylvisaker logistic regression fit.

    Attributes
    ----------
    betas : FloatArray
        Estimated coefficient vector of shape (n_features, 1).
    linear_predictors : FloatArray
        The fitted linear predictors (eta = X*beta) from the model.
    mus : FloatArray
        The fitted probabilities of shape (n_samples, 1).
    y_adjusted : FloatArray
        The adjusted or true binary response vector (y).
    x_validated : FloatArray
        The validated design matrix of shape (n_samples, n_features).
    alpha : float
        Prior shrinkage hyperparameter used in the model fit to adjust responses, in the interval [0, 1].
    theta_hat : float | None
        The estimated scalar intercept coefficient from the fitted model.
        Set to None if the model is fitted without an intercept.
    """

    betas: FloatArray
    linear_predictors: FloatArray
    mus: FloatArray
    y_adjusted: FloatArray
    x_validated: FloatArray
    alpha: float
    theta_hat: float | None


@dataclass
class LogisticRegressionResult:
    """
    A dataclass to hold the results of the logistic regression fit.

    Attributes
    ----------
    betas : FloatArray
        Estimated coefficient vector of shape (n_features, 1).
    mus : FloatArray
        The fitted probabilities of shape (n_samples, 1).
    linear_predictors : FloatArray
        The fitted linear predictors (eta = X*beta) from the model.
    """

    betas: FloatArray
    mus: FloatArray
    linear_predictors: FloatArray
