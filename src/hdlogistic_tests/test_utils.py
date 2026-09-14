import numpy as np
import pytest

from hdlogistic.utils import get_intercept_idx, has_constant_col


class TestHasConstantCol:
    def test_detects_intercept_first_column(self):
        x = np.array([[1.0, 0.5], [1.0, -0.3], [1.0, 1.1]])
        assert has_constant_col(x) is True

    def test_detects_intercept_middle_column(self):
        x = np.array([[0.5, 1.0, -1.2], [-0.3, 1.0, 0.8], [1.1, 1.0, 0.2]])
        assert has_constant_col(x) is True

    def test_no_constant_column(self):
        x = np.array([[0.5, -1.2], [-0.3, 0.8], [1.1, 0.2]])
        assert has_constant_col(x) is False

    def test_all_columns_constant(self):
        x = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        assert has_constant_col(x) is True

    def test_single_row_matrix_is_trivially_constant(self):
        # Edge case: with n=1, every column is "constant" by definition
        x = np.array([[0.5, -1.2, 3.0]])
        assert has_constant_col(x) is True


class TestGetInterceptIdx:
    def test_intercept_at_index_zero(self):
        x = np.array([[1.0, 0.5], [1.0, -0.3], [1.0, 1.1]])
        assert get_intercept_idx(x) == 0

    def test_intercept_at_middle_index(self):
        x = np.array([[0.5, 1.0, -1.2], [-0.3, 1.0, 0.8], [1.1, 1.0, 0.2]])
        assert get_intercept_idx(x) == 1

    def test_intercept_at_last_index(self):
        x = np.array([[0.5, -1.2, 1.0], [-0.3, 0.8, 1.0], [1.1, 0.2, 1.0]])
        assert get_intercept_idx(x) == 2

    def test_no_constant_column_returns_none(self):
        x = np.array([[0.5, -1.2], [-0.3, 0.8], [1.1, 0.2]])
        assert get_intercept_idx(x) is None

    def test_returns_first_constant_column_when_multiple_exist(self):
        x = np.array([[1.0, 0.5, 5.0], [1.0, -0.3, 5.0], [1.0, 1.1, 5.0]])
        assert get_intercept_idx(x) == 0


class TestHasConstantColAgreesWithGetInterceptIdx:
    """These two functions must always agree on
    whether a constant column exists."""

    @pytest.mark.parametrize(
        "x",
        [
            np.array([[1.0, 0.5], [1.0, -0.3], [1.0, 1.1]]),
            np.array([[0.5, -1.2], [-0.3, 0.8], [1.1, 0.2]]),
            np.array([[0.5, 1.0, -1.2], [-0.3, 1.0, 0.8], [1.1, 1.0, 0.2]]),
            np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
        ],
    )
    def test_consistency(self, x):
        has_const = has_constant_col(x)
        idx = get_intercept_idx(x)
        assert has_const == (idx is not None), (
            f"_has_constant_col={has_const} but _get_intercept_idx={idx} "
            "-- these two functions disagree on the same input."
        )
