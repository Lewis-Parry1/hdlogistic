from __future__ import annotations

from functools import cached_property
from typing import Any, NoReturn

import numpy as np
from scipy.stats import norm
from statsmodels.base.model import (  # pyright: ignore[reportMissingTypeStubs]
    Model,
    Results,
)
from statsmodels.iolib.summary import Summary  # pyright: ignore[reportMissingTypeStubs]

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
    def tvalues(self) -> FloatArray:
        """Alias for statsmodels Summary table generation."""
        return self.zvalues

    @property
    def pvalues(self) -> FloatArray:
        return self._summary_data.pvalues

    def conf_int(self, alpha: float = 0.05, cols: Any = None) -> FloatArray:
        """Constructs two-sided normal confidence intervals for statsmodels Summary."""
        q = norm.ppf(1.0 - alpha / 2.0)
        lower = self.params - q * self.bse
        upper = self.params + q * self.bse
        ci = np.column_stack((lower, upper))
        return ci[cols] if cols is not None else ci

    @property
    def scale(self) -> float:
        return 1.0

    @property
    def df_resid(self) -> float:
        if hasattr(self.model, "df_resid"):
            return float(self.model.df_resid)
        return float(self.nobs - len(self.params))

    @property
    def df_model(self) -> float:
        if hasattr(self.model, "df_model"):
            return float(self.model.df_model)
        return float(len(self.params) - 1)

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

    # ------------------------------------------------------------------
    # Summary & string representation
    # ------------------------------------------------------------------
    def summary(
        self,
        yname: str | None = None,
        xname: list[str] | None = None,
        title: str | None = None,
        alpha: float = 0.05,
    ) -> Summary:
        """
        Summarize the Regression Results.

        Parameters
        ----------
        yname : str, optional
            Name of the dependent variable. Default is inferred from `model.endog_names`.
        xname : list[str], optional
            Names for the exogenous variables. Must match parameter length.
        title : str, optional
            Title for the top table.
        alpha : float, default 0.05
            Significance level for the confidence intervals.

        Returns
        -------
        smry : statsmodels.iolib.summary.Summary
            Summary instance containing formatted tables and diagnostic notes.
        """
        model = self.model
        raw = self._raw_results

        if yname is None:
            yname = getattr(model, "endog_names", "y")
        if xname is None:
            xname = getattr(model, "exog_names", None)
            if xname is None:
                xname = [f"x{i}" for i in range(len(self.params))]

        if title is None:
            title = (
                "MDYPL Logistic Regression Results (HD-corrected)"
                if self.use_hd_correction
                else "MDYPL Logistic Regression Results"
            )

        method = getattr(model, "method", getattr(raw, "method", "IRLS"))

        top_left = [
            ("Dep. Variable:", [yname]),
            ("Model:", ["MDYPL-GLM"]),
            ("Method:", [str(method)]),
            ("Scale:", [f"{self.scale:#8.5g}"]),
            ("No. Iterations:", [str(self.iterations)]),
        ]

        nobs_str = f"{int(self.nobs)}" if float(self.nobs).is_integer() else f"{self.nobs:.2f}"

        top_right = [
            ("No. Observations:", [nobs_str]),
            ("Df Residuals:", [f"{int(self.df_resid)}"]),
            ("Df Model:", [f"{int(self.df_model)}"]),
            ("Deviance:", [f"{self.deviance:#8.5g}"]),
            ("AIC:", [f"{self.aic:#8.5g}"]),
            ("BIC:", [f"{self.bic:#8.5g}"]),
        ]

        smry = Summary()
        smry.add_table_2cols(
            self,
            gleft=top_left,
            gright=top_right,
            yname=yname,
            xname=xname,
            title=title,
        )
        smry.add_table_params(
            self,
            yname=yname,
            xname=xname,
            alpha=alpha,
            use_t=False,
        )

        extra_txt: list[str] = []
        if self.use_hd_correction:
            extra_txt.append("High Dimensionality Correction applied:")
            diag = self.hd_diagnostics
            sd = self._summary_data

            kappa = getattr(diag, "kappa", getattr(sd, "kappa", None))
            if kappa is not None:
                extra_txt.append(f"  Dimensionality parameter (kappa)   = {float(kappa):.3f}")

            gamma2 = getattr(
                diag,
                "signal_strength",
                getattr(sd, "signal_strength", getattr(diag, "gamma2", None)),
            )
            if gamma2 is not None:
                extra_txt.append(f"  Estimated signal strength (gamma^2) = {float(gamma2):.3f}")

            se_params = getattr(diag, "se_params", getattr(sd, "se_params", None))
            if se_params is not None:
                try:
                    formatted_tuple = f"({', '.join(f'{float(v):.3f}' for v in se_params)})"
                except (TypeError, ValueError):
                    formatted_tuple = str(se_params)
                extra_txt.append(
                    f"  State evolution parameters (mu, b, sigma, theta/iota): {formatted_tuple}"
                )

        if not self.converged:
            extra_txt.append("WARNING: The algorithm failed to converge.")

        if extra_txt:
            smry.add_extra_txt(extra_txt)

        return smry

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
