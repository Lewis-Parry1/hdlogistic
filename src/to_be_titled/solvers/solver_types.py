from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class LogisticRegressionResult:
    """
    A dataclass to hold the results of the logistic regression fit.

    Attributes
    ----------
    betas : NDArray[np.float64]
        Estimated coefficient vector of shape (n_features, 1).
    mus : NDArray[np.float64]
        The fitted probabilities of shape (n_samples, 1).
    linear_predictors : NDArray[np.float64]
        The fitted linear predictors (eta = X*beta) from the model.
    """

    betas: NDArray[np.float64]
    mus: NDArray[np.float64]
    linear_predictors: NDArray[np.float64]


@dataclass
class StateParameters:
    mu: float
    b: float
    sigma: float
    iota: float | None

    def to_array(self) -> NDArray[np.float64]:
        if self.iota is None:
            return np.array([self.mu, self.b, self.sigma])
        else:
            return np.array([self.mu, self.b, self.sigma, self.iota])


@dataclass
class SolverResult:
    """
    Container for the outcome of a numerical optimisation or root-finding
    procedure.

    Attributes
    ----------
    solution, StateParameters
        Estimated state evolution parameters.

    func_value, NDArray[np.float64]
        Residual vector evaluated at ``solution``.

    message, str
        Solver termination message.

    success, bool
        Whether the solver reported successful convergence.
    """

    solution: StateParameters
    func_value: NDArray[np.float64]
    message: str
    success: bool
