# pyright: reportPrivateUsage=false
from typing import Any

import numpy as np
import pytest

from to_be_titled import estimation


def test_fit_diaconis_ylvisaker_logistic_regression_balanced_dataset() -> None:
    x = np.ones((4, 1), dtype=np.float64)
    y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)

    result = estimation.fit_diaconis_ylvisaker_logistic_regression(
        x, y, alpha=1.0, maxiter=100, tol=1e-8
    )
    betas = result.betas

    assert betas.shape == (1, 1)
    np.testing.assert_allclose(betas, np.zeros_like(betas), atol=1e-4)


def test_fit_diaconis_ylvisaker_logistic_regression_returns_finite_values() -> None:
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)

    result = estimation.fit_diaconis_ylvisaker_logistic_regression(
        x, y, alpha=0.5, maxiter=100, tol=1e-8
    )
    betas = result.betas

    assert betas.shape == (1, 1)
    assert np.isfinite(betas).all()


def test_fit_diaconis_ylvisaker_logistic_regression_shrinks_coefficients() -> None:
    x = np.ones((3, 1), dtype=np.float64)
    y = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)

    betas_mle = estimation.fit_diaconis_ylvisaker_logistic_regression(
        x, y, alpha=1.0, maxiter=100, tol=1e-8
    ).betas
    betas_shrunk = estimation.fit_diaconis_ylvisaker_logistic_regression(
        x, y, alpha=0.5, maxiter=100, tol=1e-8
    ).betas

    assert betas_mle.shape == betas_shrunk.shape
    assert abs(betas_shrunk[0, 0]) < abs(betas_mle[0, 0])
    assert betas_shrunk[0, 0] > betas_mle[0, 0]


def test_fit_diaconis_ylvisaker_logistic_regression_accepts_flat_responses() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    result = estimation.fit_diaconis_ylvisaker_logistic_regression(
        x, y, alpha=0.5, maxiter=100, tol=1e-8
    )
    betas = result.betas

    assert betas.shape == (1, 1)
    assert np.isfinite(betas).all()


def test_fit_diaconis_ylvisaker_logistic_regression_rejects_invalid_alpha() -> None:
    x = np.ones((3, 1), dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        estimation.fit_diaconis_ylvisaker_logistic_regression(x, y, alpha=1.5)


def test_fit_diaconis_ylvisaker_logistic_regression_forwards_fit_kwargs() -> None:
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    fit_kwargs: dict[str, Any] = {"cov_type": "nonrobust"}

    result = estimation.fit_diaconis_ylvisaker_logistic_regression(
        x, y, alpha=0.5, fit_kwargs=fit_kwargs
    )

    assert result.betas.shape == (1, 1)
    assert np.isfinite(result.betas).all()
