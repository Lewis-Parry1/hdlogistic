import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from to_be_titled.solvers.logistic_regression_nesterov_gd_solver import (
    fit_logistic_regression_nesterov_accelerated_gradient_descent,
)
from to_be_titled.types import LogisticRegressionResult


def test_fit_logistic_regression_nagd_single_iteration_success() -> None:
    x = np.array(
        [
            [1.0, 2.0],
            [1.0, -1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([[1.0], [0.0]], dtype=np.float64)

    # Execute exactly 1 iteration
    result = fit_logistic_regression_nesterov_accelerated_gradient_descent(
        x=x,
        y=y,
        max_iterations=1,
        learning_rate=0.1,
    )

    expected_betas = np.array([[0.0], [0.075]], dtype=np.float64)
    expected_etas = np.array([[0.15], [-0.075]], dtype=np.float64)
    expected_mus = expit(expected_etas)

    assert isinstance(result, LogisticRegressionResult)
    assert_allclose(result.betas, expected_betas, rtol=1e-7, atol=1e-10)
    assert_allclose(result.linear_predictors, expected_etas, rtol=1e-7, atol=1e-10)
    assert_allclose(result.mus, expected_mus, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize(
    ("n_samples", "n_features"),
    [
        (10, 2),
        (50, 5),
        (100, 15),
    ],
)
def test_fit_logistic_regression_nagd_output_shapes_and_types(
    n_samples: int, n_features: int
) -> None:
    """Test that matrix operations maintain strict dimensionality boundaries."""
    # Create an arbitrary dataset
    rng = np.random.default_rng(seed=42)
    x = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=(n_samples, 1)).astype(np.float64)

    result = fit_logistic_regression_nesterov_accelerated_gradient_descent(
        x=x,
        y=y,
        max_iterations=5,  # Keep low for shape testing
    )

    # Validate output payload packaging and dimensions
    assert isinstance(result, LogisticRegressionResult)

    assert result.betas.shape == (n_features, 1)
    assert result.betas.dtype == np.float64

    assert result.linear_predictors.shape == (n_samples, 1)
    assert result.linear_predictors.dtype == np.float64

    assert result.mus.shape == (n_samples, 1)
    assert result.mus.dtype == np.float64


def test_fit_logistic_regression_nagd_early_stopping_on_convergence() -> None:
    """Test that the loop breaks early if the convergence tolerance is met."""
    x = np.array([[1.0, 1.0], [1.0, -1.0]])
    y = np.array([[1.0], [0.0]])

    # Run with an extremely high tolerance to force immediate convergence
    result_high_tol = fit_logistic_regression_nesterov_accelerated_gradient_descent(
        x=x, y=y, max_iterations=500, tolerance=10.0, learning_rate=0.1
    )

    # Run with standard tolerance
    result_std_tol = fit_logistic_regression_nesterov_accelerated_gradient_descent(
        x=x, y=y, max_iterations=500, tolerance=1e-6, learning_rate=0.1
    )

    # The high tolerance run should exit instantly (after 1 step), yielding different
    # betas than the standard run that optimizes further.
    with pytest.raises(AssertionError):
        assert_allclose(result_high_tol.betas, result_std_tol.betas)


# TODO: Add tests for edge cases like singular matrices and extreme learning rates
# TODO: Add golden master tests with expected outputs for multiple iterations
# TODO: Add intergration test comparing this against our other solvers and the R one
