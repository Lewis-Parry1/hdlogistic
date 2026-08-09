from collections.abc import Callable
from typing import TypeVar

from to_be_titled.solvers.solver_types import SolverFunction

SOLVERS_REGISTRY: dict[str, SolverFunction] = {}

F = TypeVar("F", bound=SolverFunction)


def register_solver(name: str) -> Callable[[F], F]:
    """Decorator to register a solver function under a string key."""

    def decorator(func: F) -> F:
        key = str(name).lower()
        if key in SOLVERS_REGISTRY:
            raise ValueError(f"Solver '{key}' is already registered.")
        SOLVERS_REGISTRY[key] = func
        return func

    return decorator
