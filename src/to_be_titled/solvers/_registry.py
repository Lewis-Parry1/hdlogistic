import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

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


def resolve_solver(
    solver: str | SolverFunction,
    solver_kwargs: dict[str, Any] | None = None,
) -> SolverFunction:
    """Resolves string/callable into an execution-ready solver and validates kwargs."""
    if not isinstance(solver, str) and not callable(solver):
        raise TypeError(
            f"`solver` must be a string or Callable, got {type(solver).__name__}"
        )

    if callable(solver):
        if solver_kwargs:
            raise ValueError(
                "`solver_kwargs` cannot be provided when `solver` is a "
                "pre-configured callable. "
                "Pass arguments directly via `functools.partial` instead."
            )
        return solver

    solver_key = solver.lower().strip()
    if solver_key not in SOLVERS_REGISTRY:
        valid_solvers = ", ".join(f"'{s}'" for s in SOLVERS_REGISTRY.keys())
        raise ValueError(
            f"Unknown solver '{solver}'. Available solvers: [{valid_solvers}]"
        )

    solver_function = SOLVERS_REGISTRY[solver_key]
    kwargs = solver_kwargs or {}

    if kwargs:
        try:
            solver_function_signature = inspect.signature(solver_function)
            solver_function_signature.bind_partial(**kwargs)
        except ValueError:
            pass
        except TypeError as e:
            raise TypeError(f"Invalid arguments for solver '{solver}': {e}") from None

        return partial(solver_function, **kwargs)

    return solver_function
