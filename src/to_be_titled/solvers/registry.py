from enum import StrEnum

from to_be_titled.models import (
    FisherScoringConfig,
    NAGDConfig,
    SolverEndpoint,
)
from to_be_titled.solvers.logistic_regression_fisher_solver import (
    fit_logistic_regression_fisher_scoring,
)
from to_be_titled.solvers.logistic_regression_nesterov_gd_solver import (
    fit_logistic_regression_nesterov_accelerated_gradient_descent,
)


class SolverKind(StrEnum):
    FISHER_SCORING = "fisher_scoring"
    NESTEROV_GRADIENT_DESCENT = "nesterov_gradient_descent"


SOLVERS_REGISTRY: dict[SolverKind, SolverEndpoint] = {
    SolverKind.FISHER_SCORING: SolverEndpoint(
        method=fit_logistic_regression_fisher_scoring,
        config_schema=FisherScoringConfig,
    ),
    SolverKind.NESTEROV_GRADIENT_DESCENT: SolverEndpoint(
        method=fit_logistic_regression_nesterov_accelerated_gradient_descent,
        config_schema=NAGDConfig,
    ),
}
