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
    x = np.array([[1.0, 0.0]], dtype=np.float64)
    betas = np.array([[0.5], [0.5]], dtype=np.float64)

    with pytest.raises(
        ValueError,
        match=r"Invalid prediction type 'invalid'\. Expected 'response' or 'link'\.",
    ):
        predict(x, betas, type="invalid")  # type: ignore[arg-type]


def test_predict_rejects_feature_mismatch() -> None:
    x = np.ones((5, 3), dtype=np.float64)
    betas = np.ones((2, 1), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match=(
            r"Number of coefficients \(2\) does not match number of "
            r"features in x \(3\)\."
        ),
    ):
        predict(x, betas)


def test_predict_rejects_offset_sample_mismatch() -> None:
    x = np.ones((5, 2), dtype=np.float64)
    betas = np.ones((2, 1), dtype=np.float64)
    offset = np.ones((3, 1), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match=r"Offset length \(3\) does not match number of samples in x \(5\)\.",
    ):
        predict(x, betas, offset=offset)


def test_predict_rejects_invalid_array_dimensions() -> None:
    # 3-D design matrix
    with pytest.raises(ValueError, match=r"x must be a 2-D design matrix\."):
        predict(np.ones((2, 2, 2)), np.ones((2, 1)))

    # Multi-column coefficient matrix
    with pytest.raises(
        ValueError, match=r"betas must be a 1-D array or 2-D column vector\."
    ):
        predict(np.ones((2, 2)), np.ones((2, 2)))

    # Multi-column offset matrix
    with pytest.raises(
        ValueError, match=r"offset must be a 1-D array or 2-D column vector\."
    ):
        predict(np.ones((2, 2)), np.ones((2, 1)), offset=np.ones((2, 2)))


def test_predict_with_offset_link_and_response_scales() -> None:
    x = np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    betas = np.array([[0.0], [0.5]], dtype=np.float64)

    offset = np.array([[1.0], [-0.5]], dtype=np.float64)

    expected_eta = np.array([[1.0], [0.5]], dtype=np.float64)
    eta = predict(x, betas, offset=offset, type="link")

    assert eta.shape == (2, 1)
    assert eta.dtype == np.float64
    np.testing.assert_allclose(eta, expected_eta, rtol=1e-12)

    expected_probs = np.array(
        [[0.7310585786300049], [0.6224593312018546]], dtype=np.float64
    )
    probs = predict(x, betas, offset=offset, type="response")

    assert probs.shape == (2, 1)
    assert probs.dtype == np.float64
    np.testing.assert_allclose(probs, expected_probs, rtol=1e-12)
