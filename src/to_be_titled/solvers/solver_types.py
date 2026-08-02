from dataclasses import dataclass

import numpy as np

from to_be_titled.types import FloatArray


@dataclass
class StateParameters:
    mu: float
    b: float
    sigma: float
    iota: float | None
    theta: float | None

    def to_array(self) -> FloatArray:
        arr = [self.mu, self.b, self.sigma]
        if self.iota is not None: 
            arr.append(self.iota)
        elif self.theta is not None: 
            arr.append(self.theta)
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
