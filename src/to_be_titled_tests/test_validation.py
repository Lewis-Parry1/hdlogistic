import numpy as np

from to_be_titled.validation import is_full_rank



class TestObviousCases:
    def test_identity_matrix_is_full_rank(self):
        assert is_full_rank(np.eye(5)) is True

    def test_random_matrix_is_full_rank(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((20, 5))
        assert is_full_rank(x) is True

    def test_exact_duplicate_column_is_rank_deficient(self):
        x = np.array(
            [
                [1.0, 2.0, 1.0],
                [2.0, 1.0, 2.0],
                [3.0, 0.5, 3.0],
                [4.0, -1.0, 4.0],
            ]
        )
        assert is_full_rank(x) is False

    def test_all_zero_column_is_rank_deficient(self):
        x = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )
        assert is_full_rank(x) is False

    def test_all_zero_matrix_is_rank_deficient(self):
        assert is_full_rank(np.zeros((5, 3))) is False


class TestHardCases:
    def test_exact_linear_combination_is_rank_deficient(self):
        # column 2 = column 0 + column 1
        rng = np.random.default_rng(2)
        col0 = rng.standard_normal(10)
        col1 = rng.standard_normal(10)
        x = np.column_stack([col0, col1, col0 + col1])
        assert is_full_rank(x) is False

    def test_dummy_variable_trap_is_rank_deficient(self):
        # intercept + all category dummies (no reference level dropped)
        intercept = np.ones(6)
        dummy_a = np.array([1, 0, 0, 1, 0, 0], dtype=float)
        dummy_b = np.array([0, 1, 0, 0, 1, 0], dtype=float)
        dummy_c = np.array([0, 0, 1, 0, 0, 1], dtype=float)
        x = np.column_stack([intercept, dummy_a, dummy_b, dummy_c])
        assert is_full_rank(x) is False

    def test_wide_scale_range_stays_full_rank(self):
        # columns spanning many orders of magnitude
        rng = np.random.default_rng(11)
        n, p = 200, 8
        scales = 10.0 ** np.arange(-4, 4)
        x = rng.standard_normal((n, p)) * scales
        assert is_full_rank(x) is True

    def test_more_columns_than_rows_is_rank_deficient(self):
        # p > n: never full rank, regardless of the actual entries
        rng = np.random.default_rng(5)
        x = rng.standard_normal((3, 6))
        assert is_full_rank(x) is False

    def test_tiny_machine_precision_perturbation_stays_full_rank(self):
        # perturbation larger than qr_tol
        # -- should not be flagged as deficient
        rng = np.random.default_rng(3)
        col0 = rng.standard_normal(20)
        col1 = col0 + 1e-6 * rng.standard_normal(20)
        x = np.column_stack([col0, col1])
        assert is_full_rank(x, qr_tol=1e-11) is True

    def test_clearly_near_singular_column_is_flagged(self):
        # perturbation smaller than qr_tol
        # -- should be flagged as deficient
        rng = np.random.default_rng(3)
        col0 = rng.standard_normal(20)
        col1 = col0 + 1e-15 * rng.standard_normal(20)
        x = np.column_stack([col0, col1])
        assert is_full_rank(x, qr_tol=1e-11) is False

    def test_custom_tol_changes_classification_at_the_margin(self):
        # same matrix, tighter vs looser tol should disagree at a
        # deliberately marginal perturbation size
        rng = np.random.default_rng(3)
        col0 = rng.standard_normal(20)
        col1 = col0 + 1e-8 * rng.standard_normal(20)
        x = np.column_stack([col0, col1])

        assert is_full_rank(x, qr_tol=1e-11) is True
        assert is_full_rank(x, qr_tol=1e-7) is False

    def test_single_zero_column_is_rank_deficient(self):
        assert is_full_rank(np.zeros((5, 1))) is False

    def test_single_nonzero_column_is_full_rank(self):
        x = np.array([[1.0], [2.0], [3.0]])
        assert is_full_rank(x) is True

    def test_large_matrix_with_known_deficiency_via_broadcasting(self):
        rng = np.random.default_rng(42)
        n, p_independent, p_redundant = 500, 30, 5

        x_independent = rng.standard_normal((n, p_independent))
        mixing = rng.standard_normal((p_independent, p_redundant))

        x_redundant = np.zeros((n, p_redundant))
        for j in range(p_redundant):
            x_redundant[:, j] = (x_independent * mixing[:, j]).sum(axis=1)

        x = np.column_stack([x_independent, x_redundant])
        assert is_full_rank(x) is False

    def test_large_full_rank_matrix(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((1000, 50))
        assert is_full_rank(x) is True


class TestReturnType:
    def test_returns_python_bool(self):
        result = is_full_rank(np.eye(3))
        assert isinstance(result, bool)
