import numpy as np
from numpy.testing import assert_allclose

from to_be_titled.inference import compute_sloe_estimator


def test_compute_sloe_estimator_no_leverage_adjustment() -> None:
    linear_predictors = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    y_adjusted = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    leverages = np.zeros(3, dtype=np.float64)

    expected = float(np.std(linear_predictors, ddof=1))
    assert_allclose(
        compute_sloe_estimator(linear_predictors, y_adjusted, leverages),
        expected,
    )


def test_compute_sloe_estimator_excludes_infinite_values() -> None:
    linear_predictors = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    y_adjusted = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    leverages = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    expected = float(np.std(np.array([0.0, -1.0], dtype=np.float64), ddof=1))
    assert_allclose(
        compute_sloe_estimator(linear_predictors, y_adjusted, leverages),
        expected,
    )


def test_compute_sloe_estimator_happy_path() -> None:
    """Verify SLOE calculation with standard inputs."""

    # Setup inputs where 0 < mu < 1 and 0 < h < 1
    linear_predictors = np.array([0.0, 0.5, -0.5], dtype=np.float64)
    y_adjusted = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    leverages = np.array([0.2, 0.3, 0.1], dtype=np.float64)

    # Expected sample standard deviation of S with ddof=1
    expected_signal_strength = 1.3258861214054045

    assert_allclose(
        compute_sloe_estimator(linear_predictors, y_adjusted, leverages),
        expected_signal_strength,
        rtol=2e-6,
    )


# TODO: Move these tests into a separate file for testing interface
