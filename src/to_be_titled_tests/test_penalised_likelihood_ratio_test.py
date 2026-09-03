import numpy as np
import pytest
from to_be_titled.mdypl_fit import MDYPLModel

from to_be_titled.penalised_likelihood_ratio_test import (
    PenalisedLRTResults,
    penalised_lrt,
)
from to_be_titled.types import MDYPLResults

x = np.array(
    [
        [1.0, 0.5, -1.2],
        [1.0, -0.3, 0.8],
        [1.0, 1.1, 0.2],
        [1.0, -0.7, -0.5],
        [1.0, 0.2, 1.4],
        [1.0, 1.5, -0.9],
        [1.0, -1.1, 0.3],
        [1.0, 0.8, 0.6],
        [1.0, -0.4, -1.3],
        [1.0, 0.6, 0.1],
    ]
)
y = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.float64)


@pytest.fixture(scope="module")
def full_result() -> MDYPLResults:
    """Full model: intercept + both covariates."""
    model = MDYPLModel(y=y, x=x)
    return model.fit()


@pytest.fixture(scope="module")
def reduced_result(full_result: MDYPLResults) -> MDYPLResults:
    """Reduced model: intercept + first covariate only.

    Uses the full model's alpha explicitly so the two fits are
    comparable (see penalised_lrt's alpha-mismatch check).
    """
    model = MDYPLModel(y=y, x=x[:, :2], alpha=full_result.alpha)
    return model.fit()


class TestPenalisedLRTTypesAndShapes:
    def test_returns_penalised_lrt_results(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=False)
        assert isinstance(result, PenalisedLRTResults)

    @pytest.mark.parametrize("hd_correction", [False, True])
    def test_statistic_is_float(
        self,
        full_result: MDYPLResults,
        reduced_result: MDYPLResults,
        hd_correction: bool,
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=hd_correction)
        assert isinstance(result.statistic, float)
        assert np.isfinite(result.statistic)

    @pytest.mark.parametrize("hd_correction", [False, True])
    def test_p_value_is_valid_probability(
        self,
        full_result: MDYPLResults,
        reduced_result: MDYPLResults,
        hd_correction: bool,
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=hd_correction)
        assert isinstance(result.p_value, float)
        assert 0.0 <= result.p_value <= 1.0

    def test_df_matches_rank_difference(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=False)
        assert result.df == full_result.rank - reduced_result.rank
        assert isinstance(result.df, (int, np.integer))

    def test_argument_order_does_not_matter(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        """penalised_lrt should auto-detect full vs. reduced by rank."""
        result_fwd = penalised_lrt(full_result, reduced_result, hd_correction=False)
        result_rev = penalised_lrt(reduced_result, full_result, hd_correction=False)
        assert result_fwd.statistic == pytest.approx(result_rev.statistic)
        assert result_fwd.p_value == pytest.approx(result_rev.p_value)
        assert result_fwd.df == result_rev.df

    def test_hd_correction_false_leaves_diagnostics_none(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=False)
        assert result.kappa is None
        assert result.se_params is None
        assert result.signal_strength is None

    def test_hd_correction_true_populates_diagnostics(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=True)
        assert isinstance(result.kappa, float)
        assert result.se_params is not None
        assert result.se_params.shape[0] >= 3  # at least (mu, b, sigma)
        assert isinstance(result.signal_strength, float)

    def test_dict_style_access_matches_attributes(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=True)
        assert result["statistic"] == result.statistic
        assert result["p_value"] == result.p_value
        assert result["df"] == result.df
        assert result["kappa"] == result.kappa

    def test_summary_returns_string(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=True)
        assert isinstance(str(result), str)
        assert isinstance(repr(result), str)


class TestPenalisedLRTValidation:
    def test_raises_on_alpha_mismatch(self, full_result: MDYPLResults) -> None:
        mismatched = MDYPLModel(y=y, x=x[:, :2], alpha=0.1).fit()
        with pytest.raises(ValueError, match="alpha"):
            penalised_lrt(full_result, mismatched)

    def test_raises_on_different_response(self, full_result: MDYPLResults) -> None:
        y_other = y.copy()
        y_other[0] = 1.0 - y_other[0]
        other = MDYPLModel(y=y_other, x=x[:, :2], alpha=full_result.alpha).fit()
        with pytest.raises(ValueError, match="response"):
            penalised_lrt(full_result, other)


class TestPenalisedLRTAgainstBrglm2:
    """Reference-value comparison against brglm2 output on the same
    fixed dataset. Fill in `EXPECTED_*` once brglm2 results are available.
    """

    EXPECTED_STATISTIC_NO_HD = 0.114277032582484
    EXPECTED_PVALUE_NO_HD = 0.735326366983393
    EXPECTED_STATISTIC_HD = 0.199078055788034
    EXPECTED_PVALUE_HD = 0.655466044162702

    def test_matches_brglm2_no_hd_correction(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=False)
        assert result.statistic == pytest.approx(
            self.EXPECTED_STATISTIC_NO_HD, rel=1e-6
        )
        assert result.p_value == pytest.approx(self.EXPECTED_PVALUE_NO_HD, rel=1e-6)

    def test_matches_brglm2_hd_correction(
        self, full_result: MDYPLResults, reduced_result: MDYPLResults
    ) -> None:
        result = penalised_lrt(full_result, reduced_result, hd_correction=True)
        assert result.statistic == pytest.approx(self.EXPECTED_STATISTIC_HD, rel=1e-6)
        assert result.p_value == pytest.approx(self.EXPECTED_PVALUE_HD, rel=1e-6)
