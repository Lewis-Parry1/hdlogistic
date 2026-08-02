from unittest.mock import MagicMock

import numpy as np
import pytest

from to_be_titled.solvers.solver_types import SolverResult, StateParameters
from to_be_titled.summary import (
    _compute_leverages,  # pyright: ignore [reportPrivateUsage]
    summary,
)
from to_be_titled.types import DiaconisYlvisakerLogisticRegressionResult


def _make_result(
    n_samples: int = 10,
    n_features: int = 3,
    seed: int = 0,
) -> DiaconisYlvisakerLogisticRegressionResult:
    """Helper factory to construct mock estimation results."""
    x_validated = np.arange(n_samples * n_features, dtype=np.float64).reshape(
        n_samples,
        n_features,
    )
    betas = np.arange(n_features, dtype=np.float64) + 1.0
    linear_predictors = x_validated @ betas
    mus = np.expand_dims(1 / (1 + np.exp(-linear_predictors)), axis=1)
    y_adjusted = linear_predictors + 0.1

    return DiaconisYlvisakerLogisticRegressionResult(
        x_validated=x_validated,
        y_adjusted=y_adjusted,
        linear_predictors=linear_predictors,
        betas=betas,
        mus=mus,
        alpha=1.0,
        theta_hat=None,
    )


def test_compute_leverages_returns_shape_and_range() -> None:
    result = _make_result(n_samples=10, n_features=3, seed=0)
    leverages = _compute_leverages(result.x_validated, result.mus)

    assert leverages.shape == (10,)
    assert np.all(leverages >= 0)
    assert np.all(leverages <= 1.0 + 1e-8)


def test_summary_with_high_dimensional_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_result(n_samples=10, n_features=3, seed=0)

    # Mock _solve_state_equation to return a specific mu_star
    mock_mu_star = 0.5
    fake_pars = StateParameters(mock_mu_star, 1, 1, iota=None, theta=None)

    mock_solver_result = SolverResult(
        solution=fake_pars,
        func_value=np.array([0.0, 0.0, 0.0]),
        message="Success",
        success=True,
    )

    monkeypatch.setattr(
        "to_be_titled.summary.solve_state_equation",
        MagicMock(return_value=(mock_solver_result, "success")),
    )
    monkeypatch.setattr(
        "to_be_titled.summary.compute_sloe_estimator",
        MagicMock(return_value=1.5),
    )

    output = summary(result, high_dimensional_correction=True)

    expected_betas = result.betas / mock_mu_star
    assert np.allclose(output, expected_betas)


def test_summary_without_high_dimensional_correction() -> None:
    result = _make_result(n_samples=10, n_features=3, seed=0)
    output = summary(result, high_dimensional_correction=False)
    assert np.allclose(output, result.betas)


def test_summary_integration_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_result(n_samples=10, n_features=3, seed=2)
    output = summary(result, high_dimensional_correction=True)
    assert output.shape == result.betas.shape
    assert np.all(np.isfinite(output))
    assert not np.allclose(output, result.betas)
