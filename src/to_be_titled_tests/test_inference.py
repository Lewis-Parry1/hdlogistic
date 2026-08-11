import numpy as np
import pytest

from to_be_titled import predict


def test_predict_response_scale_known_values() -> None:
    # x @ betas yields eta = [[0.0], [1.0]]
    x = np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    betas = np.array([[0.0], [0.5]], dtype=np.float64)

    expected_probs = np.array([[0.5], [0.7310585786300049]], dtype=np.float64)

    probs = predict(x, betas, type="response")

    assert probs.shape == (2, 1)
    assert probs.dtype == np.float64
    np.testing.assert_allclose(probs, expected_probs, rtol=1e-12)


def test_predict_link_scale_linear_predictors() -> None:
    x = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    betas = np.array([[2.0], [0.5]], dtype=np.float64)

    expected_eta = np.array([[1.0], [8.0]], dtype=np.float64)

    eta = predict(x, betas, type="link")

    assert eta.shape == (2, 1)
    np.testing.assert_allclose(eta, expected_eta, rtol=1e-12)


def test_predict_rejects_invalid_type() -> None:
    x = np.ones((3, 2), dtype=np.float64)
    betas = np.ones((2, 1), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match=r"Invalid prediction type 'invalid'. Expected 'response' or 'link'.",
    ):
        predict(x, betas, type="invalid")  # type: ignore[arg-type]
