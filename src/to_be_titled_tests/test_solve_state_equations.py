import numpy as np
import pytest
from numpy.typing import NDArray

from to_be_titled.solve_state_equations import _solve_state_equation
from to_be_titled.utils import _get_hermite_roots_weights

"""

"""


# Compare solve_se to Candes and Sur results.
# See Table 13, set alpha = 1, when gamma = np.sqrt(5 + beta0^2)
@pytest.mark.parametrize(
    "gamma, expected_roots",
    [
        (np.sqrt(5), np.array([1.50, 3.03, 4.75])),
        (np.sqrt(6), np.array([1.55, 3.42, 5.13])),
        (np.sqrt(9), np.array([1.75, 4.83, 6.45])),
        (np.sqrt(11.25), np.array([1.95, 6.26, 7.73])),
    ],
)
def test_solve_state_equation_compare_candes_sur(
    gamma: float, expected_roots: NDArray[np.float64]
) -> None:
    kappa = 0.2
    alpha = 1.0
    gh = _get_hermite_roots_weights(200)

    # Use default brglm2 starting guess
    start = np.array([0.5, 1, 1])

    solver_result, _ = _solve_state_equation(kappa, gamma, alpha, start, gh)

    # Ensure solver converged successfully
    assert solver_result.success is True, "Solver failed to converge"
    np.testing.assert_allclose(solver_result.solution, expected_roots, atol=1e-2)


# Test against solve_state_equations against brglm2 for different kappa/gamma/alpha
