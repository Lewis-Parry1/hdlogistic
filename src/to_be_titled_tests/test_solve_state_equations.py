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
    "thetas, roots",
    [
        (0.0, np.array([1.50, 3.03, 4.75])),
        (0.5, np.array([1.51, 3.12, 4.84])),
        (1.0, np.array([1.55, 3.42, 5.13])),
        (2, np.array([1.75, 4.83, 6.45])),
        (2.5, np.array([1.95, 6.26, 7.73])),
    ],
)
def test_solve_state_equation_compare_candes_sur(
    thetas: float, roots: NDArray[np.float64]
) -> None:
    kappa = 0.2
    alpha = 1.0
    gamma = np.sqrt(5 + thetas**2)

    gh = _get_hermite_roots_weights(200)

    # Use default brglm2 starting guess
    start = np.array([0.5, gamma, gamma])

    solver_result, _ = _solve_state_equation(
        kappa, gamma, alpha, start, gh, init_iter=50
    )

    # Ensure solver converged successfully
    assert solver_result.success is True, "Solver failed to converge"
    np.testing.assert_allclose(solver_result.solution, roots, atol=1e-2)


# Test against solve_state_equations against brglm2 for different kappa/gamma/alpha
def test_solve_state_equations_against_brglm2() -> None:
    kappa, gamma, alpha = 0.2, 5, 0.88
    true_nelder_mead_50 = np.array([0.5649718, 2.5935673, 2.5376760])
    # use default, naive guess used in brglm2
    start = np.array([0.5, 1, 1])
    gh = _get_hermite_roots_weights(200)

    est_nelder_mead_50, _ = _solve_state_equation(
        kappa, gamma, alpha, start, gh, init_iter=50
    )

    np.testing.assert_allclose(
        est_nelder_mead_50.solution, true_nelder_mead_50, atol=1e-10
    )
