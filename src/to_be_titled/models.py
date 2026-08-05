from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from to_be_titled.types import LogisticRegressionResult


@dataclass(frozen=True, kw_only=True)
class BaseSolverConfig:
    """Base class for all solver configurations offering safe dictionary parsing."""

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        """Instantiate config dataclass, raising ValueError if any unrecognized keys
        are provided."""

        valid_keys = set(cls.__dataclass_fields__.keys())
        provided_keys = set(config.keys())

        unknown_keys = provided_keys - valid_keys
        if unknown_keys:
            sorted_unknown = sorted(unknown_keys)
            sorted_valid = sorted(valid_keys)
            raise ValueError(
                f"Unrecognized configuration option(s) for {cls.__name__}: "
                f"{sorted_unknown}. "
                f"Supported keys are: {sorted_valid}"
            )

        return cls(**config)


@dataclass(frozen=True, kw_only=True)
class FisherScoringConfig(BaseSolverConfig):
    """Configuration options and boundary checks for Fisher Scoring solver."""

    max_iterations: int = 25
    tolerance: float = 1e-6
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations <= 0
        ):
            raise ValueError(
                "FisherScoringConfig 'max_iterations' must be a positive integer, "
                f"got {self.max_iterations!r}"
            )
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (float, int, np.floating, np.integer))
            or self.tolerance <= 0.0
        ):
            raise ValueError(
                "FisherScoringConfig 'tolerance' must be a positive float, "
                f"got {self.tolerance!r}"
            )
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (float, int, np.floating, np.integer))
            or self.epsilon <= 0.0
        ):
            raise ValueError(
                "FisherScoringConfig 'epsilon' must be a positive float, "
                f"got {self.epsilon!r}"
            )


@dataclass(frozen=True, kw_only=True)
class NAGDConfig(BaseSolverConfig):
    """Configuration options and boundary checks for Nesterov Accelerated
    Gradient Descent solver."""

    max_iterations: int = 500
    tolerance: float = 1e-6
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations <= 0
        ):
            raise ValueError(
                "NAGDConfig 'max_iterations' must be a positive integer, "
                f"got {self.max_iterations!r}"
            )
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (float, int, np.floating, np.integer))
            or self.tolerance <= 0.0
        ):
            raise ValueError(
                "NAGDConfig 'tolerance' must be a positive float, "
                f"got {self.tolerance!r}"
            )
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (float, int, np.floating, np.integer))
            or self.learning_rate <= 0.0
        ):
            raise ValueError(
                "NAGDConfig 'learning_rate' must be a positive float, "
                f"got {self.learning_rate!r}"
            )


@dataclass(frozen=True)
class SolverEndpoint:
    """Pairs a solver function with its specific configuration schema."""

    method: Callable[..., LogisticRegressionResult]
    config_schema: type[BaseSolverConfig]
