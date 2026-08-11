from to_be_titled.solvers._registry import (
    register_solver,
)
from to_be_titled.solvers.logistic_regression_fisher_solver import (
    fit_logistic_regression_fisher_scoring,
)
from to_be_titled.solvers.logistic_regression_nesterov_gd_solver import (
    fit_logistic_regression_nagd,
)
from to_be_titled.solvers.state_equations_solver import solve_state_equation

__all__ = [
    "register_solver",
    "fit_logistic_regression_fisher_scoring",
    "fit_logistic_regression_nagd",
    "solve_state_equation",
]

# TODO: Split off logistic regression solvers into a separate submodule
