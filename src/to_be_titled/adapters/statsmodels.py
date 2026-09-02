from __future__ import annotations

import numpy as np

from functools import cached_property
from typing import Any, NoReturn

from statsmodels.base.model import (  # pyright: ignore[reportMissingTypeStubs]
    Model,
    Results,
)

from to_be_titled.estimation import (
    MDYPLData,
    MDYPLResults,
    fit_mdypl,
    prepare_mdypl_data,
)
from to_be_titled.penalised_likelihood_ratio_test import (
    PenalisedLRTResults,
    penalised_lrt,
)
from to_be_titled.summary import HDDiagnostics, MDYPLSummary, summary
from to_be_titled.types import (
    FloatArray,
)


class MDYPLLogisticResult(Results):  # type: ignore[misc]
    def __init__(
        self,
        model: Any,
        raw_results: MDYPLResults,
        use_hd_correction: bool = False,
        start: FloatArray | None = None,
        solve_se_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._raw_results: MDYPLResults = raw_results
        self.use_hd_correction: bool = use_hd_correction
        self._start: FloatArray | None = start
        self._solve_se_kwargs: dict[str, Any] | None = solve_se_kwargs

        super().__init__(model=model, params=raw_results.params, **kwargs)  # pyright: ignore[reportUnknownMemberType]

        self.converged: bool = raw_results.converged
        self.iterations: int = raw_results.iterations
        self.alpha: float = raw_results.alpha
        self.nobs: float = raw_results.nobs_eff

    #TODO: Print functionality goes here so we can have print(result) for hd_corrected
    # or not corrected result. 

   
    @cached_property
    def _summary_data(self) -> MDYPLSummary:
        return summary(
            result=self._raw_results,
            start=self._start,
            solve_se_kwargs=self._solve_se_kwargs,
            high_dimensional_correction=self.use_hd_correction,
        )

    @property
    def params(self) -> FloatArray:
        return self._summary_data.params

    @params.setter
    def params(self, value: FloatArray) -> None:
        pass

    @property
    def bse(self) -> FloatArray:
        return self._summary_data.bse

    @property
    def zvalues(self) -> FloatArray:
        return self._summary_data.zvalues

    @property
    def pvalues(self) -> FloatArray:
        return self._summary_data.pvalues

    @property
    def linear_predictors(self) -> FloatArray:
        return self._summary_data.linear_predictors

    @property
    def fitted_probs(self) -> FloatArray:
        return self._summary_data.fitted_probs

    @property
    def deviance(self) -> float:
        return self._summary_data.deviance

    @property
    def null_deviance(self) -> float:
        return self._summary_data.null_deviance

    @property
    def resid_deviance(self) -> FloatArray:
        return self._summary_data.resid_deviance

    @property 
    def resid_pearson(self) -> FloatArray:
        return self._summary_data.resid_pearson

    @property
    def aic(self) -> float:
        return self._summary_data.aic

    @property
    def bic(self) -> float:
        return self._summary_data.bic

    @property
    def hd_diagnostics(self) -> HDDiagnostics | None:
        return self._summary_data.hd_diagnostics

    @property
    def leverages(self) -> FloatArray:
        return self._raw_results.leverages

    def cov_params(
        self,
        r_matrix: Any = None,
        column: Any = None,
        scale: Any = None,
        cov_p: Any = None,
        other: Any = None,
        **kwargs: Any,
    ) -> FloatArray:
        if self.use_hd_correction:
            raise NotImplementedError(
                "Covariance matrix is undefined for high-dimensional corrected result. "
                "Use `self.bse` for rescaled standard errors."
            )
        return self._raw_results.cov_params

    def get_high_dimensional(
        self,
        start: FloatArray | None = None,
        solve_se_kwargs: dict[str, Any] | None = None,
    ) -> MDYPLLogisticResult:
        if self.use_hd_correction:
            return self

        return MDYPLLogisticResult(
            model=self.model,  # pyright: ignore[reportUnknownMemberType]
            raw_results=self._raw_results,
            use_hd_correction=True,
            start=start,
            solve_se_kwargs=solve_se_kwargs,
        )

    def penalised_lrt(
        self,
        other: MDYPLLogisticResult,
        hd_correction: bool | None = None,
        solve_se_kwargs: dict[str, Any] | None = None,
    ) -> PenalisedLRTResults | None:

            use_hd = self.use_hd_correction if hd_correction is None else hd_correction

            return penalised_lrt(
                self._raw_results,
                other._raw_results,
                hd_correction=use_hd,
                solve_se_kwargs=solve_se_kwargs,
            )


class MDYPLLogistic(Model):  # type: ignore[misc]
    def __init__(
        self,
        endog: FloatArray,
        exog: FloatArray,
        alpha: float | None = None,
        weights: FloatArray | None = None,
        offset: float | FloatArray | None = None,
        missing: str = "none",
        **kwargs: Any, 
    ) -> None:
        super().__init__(endog=endog, exog=exog, missing=missing, **kwargs)  # pyright: ignore[reportUnknownMemberType]

        self._data: MDYPLData = prepare_mdypl_data(
            x=self.exog,  # pyright: ignore[reportArgumentType, reportUnknownArgumentType, reportUnknownMemberType]
            y=self.endog,  # pyright: ignore[reportArgumentType, reportUnknownArgumentType, reportUnknownMemberType]
            weights=weights,
            offset=offset,
        )
        self.alpha = alpha
        self._init_keys.extend(["alpha", "weights", "offset"])

    def fit(
        self,
        tol: float = 1e-8,
        maxiter: int = 100,
        method: str = "IRLS",
        start_params: FloatArray | None = None,
        **kwargs: Any, 
    ) -> MDYPLLogisticResult:
        raw_results = fit_mdypl(
            data=self._data,
            alpha=self.alpha,
            tol=tol,
            maxiter=maxiter,
            method=method,
            start_params=start_params,
        )
        return MDYPLLogisticResult(model=self, raw_results=raw_results, **kwargs)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: Any,
        subset: Any = None,
        drop_cols: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        raise NotImplementedError(
            "Formula-based model specification via `from_formula` is not supported "
            "for `MDYPLLogistic`. Pass explicit `endog` and `exog` design matrices"
            " directly."
        )
