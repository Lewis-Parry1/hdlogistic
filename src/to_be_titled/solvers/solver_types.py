from dataclasses import dataclass

import numpy as np

from to_be_titled.types import FloatArray


@dataclass
class StateParameters:
    """
    Dataclass to store the estimated stationary points of MDYPL state evolution
    system of equations. When `corrupted` is False, `iota` is estimated by the
    solver, whilst when `corrupted` is True, `theta` is estimated by the solver.
    """

    mu: float
    b: float
    sigma: float
    corrupted: bool = False
    intercept_estimate: float | None = None

    @property
    def iota(self) -> float | None:
        """
        Returns `iota` if state equations solved when `corrupted` is False,
        else None.
        """
        return self.intercept_estimate if not self.corrupted else None

    @property
    def theta(self) -> float | None:
        """
        Returns `theta` if state equations solved when `corrupted` is True,
        else None.
        """
        return self.intercept_estimate if self.corrupted else None

    def to_array(self) -> FloatArray:
        """
        Returns the array of parameters estimated by the solver. When the model
        contains an intercept, `iota` is appended to the array of estimated parameters
        when using the `corrupted` is False and `theta` is appended when `corrupted`
        is True.
        """
        arr = [self.mu, self.b, self.sigma]
        if self.intercept_estimate is not None:
            arr.append(self.intercept_estimate)
        return np.array(arr, dtype=np.float64)


@dataclass
class SolverResult:
    """
    Container for the outcome of a numerical optimisation or root-finding
    procedure.

    Attributes
    ----------
    solution, StateParameters
        Estimated state evolution parameters.

    func_value, FloatArray
        Residual vector evaluated at ``solution``.

    message, str
        Solver termination message.

    success, bool
        Whether the solver reported successful convergence.
    """

    solution: StateParameters
    func_value: FloatArray
    message: str
    success: bool
