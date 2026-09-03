from __future__ import annotations

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
    """Results container for modified Diaconis–Ylvisaker penalized logistic regression.

    Encapsulates parameter estimates, standard errors, test statistics, and
    fitted values from an :class:`MDYPLLogistic` model fit. Supports standard GLM
    asymptotics as well as high-dimensional debiasing and inference via state evolution
    equations.

    Parameters
    ----------
    model : MDYPLLogistic
        The model instance that generated these results.
    raw_results : MDYPLResults
        Internal results container returned by `fit_mdypl`.
    use_hd_correction : bool, default False
        Whether high-dimensional asymptotic corrections (debiasing and
        state-evolution-derived standard errors) are active.
    start : FloatArray | None, optional
        Initial starting values `(alpha, b, sigma)` for the state evolution solver.
    solve_se_kwargs : dict[str, Any] | None, optional
        Additional solver keyword arguments forwarded to `solve_state_equation`.
    **kwargs : Any
        Additional keyword arguments passed to `statsmodels.base.model.Results`.

    Attributes
    ----------
    converged : bool
        Whether the optimization routine converged successfully.
    iterations : int
        Number of iterations executed by the optimization routine.
    alpha : float
        Shrinkage parameter value used during model fitting.
    nobs : float
        Effective sample size (sum of observation weights).
    use_hd_correction : bool
        Whether high-dimensional asymptotic corrections are applied to estimates
        and inferential statistics.
    """

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
        """Parameter estimates.

        If `use_hd_correction=True`, non-intercept coefficients are debiased
        via the state evolution scaling factor $\\hat{\\mu}$ and the intercept is
        replaced by its state evolution estimate.
        """
        return self._summary_data.params

    @params.setter
    def params(self, value: FloatArray) -> None:
        pass

    @property
    def bse(self) -> FloatArray:
        """Standard errors of the parameter estimates.

        Under standard asymptotics (`use_hd_correction=False`), standard errors
        are derived from the inverse Fisher information. Under high-dimensional
        corrections (`use_hd_correction=True`), standard errors are computed
        using state evolution parameters and probe statistics (intercept SE is
        set to NaN).
        """
        return self._summary_data.bse

    @property
    def zvalues(self) -> FloatArray:
        """Wald $z$-statistics for parameter estimates (`params / bse`)."""
        return self._summary_data.zvalues

    @property
    def pvalues(self) -> FloatArray:
        """Two-sided asymptotic p-values under the standard normal distribution."""
        return self._summary_data.pvalues

    @property
    def linear_predictors(self) -> FloatArray:
        """Linear predictors $\\eta = X\\beta + \\text{offset}$ of shape `(n,)`."""
        return self._summary_data.linear_predictors

    @property
    def fitted_probs(self) -> FloatArray:
        """Fitted response probabilities $\\mu = \\text{expit}(\\eta)$ of shape `(n,)`."""
        return self._summary_data.fitted_probs

    @property
    def deviance(self) -> float:
        """Model deviance evaluated at the current parameter estimates."""
        return self._summary_data.deviance

    @property
    def aic(self) -> float:
        """Akaike Information Criterion evaluated on the adjusted response."""
        return self._summary_data.aic

    @property
    def hd_diagnostics(self) -> HDDiagnostics | None:
        """Diagnostics and solutions from high-dimensional asymptotics, or None."""
        return self._summary_data.hd_diagnostics

    @property
    def leverages(self) -> FloatArray:
        """Leverage values (diagonal elements of the hat matrix) of shape `(n,)`."""
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
        """Estimated variance-covariance matrix of parameter estimates.

        Parameters
        ----------
        r_matrix : Any, optional
            Ignored; present for compatibility with `statsmodels.base.model.Results`.
        column : Any, optional
            Ignored; present for compatibility with `statsmodels.base.model.Results`.
        scale : Any, optional
            Ignored; present for compatibility with `statsmodels.base.model.Results`.
        cov_p : Any, optional
            Ignored; present for compatibility with `statsmodels.base.model.Results`.
        other : Any, optional
            Ignored; present for compatibility with `statsmodels.base.model.Results`.
        **kwargs : Any
            Additional keyword arguments for statsmodels compatibility.

        Returns
        -------
        FloatArray
            Covariance matrix of shape `(p, p)`.

        Raises
        ------
        NotImplementedError
            If `use_hd_correction=True`. The joint covariance matrix is undefined
            under marginal high-dimensional corrections; use :attr:`bse` instead.
        """
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
        """Return a copy of the results with high-dimensional corrections enabled.

        Solves the state evolution equations to debias coefficients and adjust
        standard errors under high-dimensional asymptotics ($p / n \\to \\kappa$).

        Parameters
        ----------
        start : FloatArray | None, optional
            Initial guess `(alpha, b, sigma)` for the state evolution solver.
        solve_se_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments forwarded to `solve_state_equation`.

        Returns
        -------
        MDYPLLogisticResult
            A new results object with `use_hd_correction=True`. Returns `self`
            unchanged if corrections are already active.
        """
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
        """Perform a penalised likelihood ratio test between two nested models.

        Parameters
        ----------
        other : MDYPLLogisticResult
            The nested or enclosing model result to compare against.
        hd_correction : bool | None, optional
            Whether to apply high-dimensional asymptotic rescaling to the test
            statistic. If None, inherits `self.use_hd_correction`, by default None.
        solve_se_kwargs : dict[str, Any] | None, optional
            Keyword arguments forwarded to `solve_state_equation` when solving
            state equations for high-dimensional rescaling, by default None.

        Returns
        -------
        PenalisedLRTResults | None
            Container storing the test statistic, degrees of freedom, p-value,
            and high-dimensional diagnostics.

        Raises
        ------
        ValueError
            If the models have different values of `alpha`, were fitted on
            different response vectors, or have identical rank.
        """
        use_hd = self.use_hd_correction if hd_correction is None else hd_correction

        return penalised_lrt(
            self._raw_results,
            other._raw_results,
            hd_correction=use_hd,
            solve_se_kwargs=solve_se_kwargs,
        )


class MDYPLLogistic(Model):  # type: ignore[misc]
    """Modified Diaconis–Ylvisaker Penalized Logistic Regression (MDYPL).

    Fits a logistic regression model with response shrinkage derived from a modified
    Diaconis–Ylvisaker conjugate prior. Binary responses $y \\in \\{0, 1\\}$ are shrunk
    toward 0.5 via shrinkage parameter $\\alpha \\in [0, 1]$:
    $\\tilde{y} = \\alpha y + (1 - \\alpha) / 2$, guaranteeing the existence of finite
    maximum penalized likelihood estimates even under complete data separation.

    Parameters
    ----------
    endog : FloatArray
        Binary response variable of shape `(n,)` or `(n, 1)`.
    exog : FloatArray
        Covariate design matrix of shape `(n, p)`.
    alpha : float | None, optional
        Shrinkage parameter constrained to `[0, 1]`. If None, automatically computed
        as `nobs_eff / (nobs_eff + rank - int(has_intercept))`, by default None.
    weights : FloatArray | None, optional
        Observation weights of shape `(n,)`. Defaults to uniform unit weights.
    offset : float | FloatArray | None, optional
        Additive offset term as a scalar or array of shape `(n,)`, by default None.
    missing : str, default "none"
        Missing value handling strategy passed to `statsmodels.base.model.Model`.
    **kwargs : Any
        Additional keyword arguments passed to the parent `Model`.

    Attributes
    ----------
    alpha : float | None
        User-specified shrinkage parameter, or None if computed dynamically during fit.
    """

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
        """Fit the model using iteratively reweighted least squares (IRLS).

        Parameters
        ----------
        tol : float, default 1e-8
            Convergence tolerance for the GLM optimization routine.
        maxiter : int, default 100
            Maximum number of iterations allowed for the solver.
        method : str, default "IRLS"
            Optimization method passed to `statsmodels.genmod.generalized_linear_model.GLM.fit`.
        start_params : FloatArray | None, optional
            Initial coefficient values of shape `(p,)` for optimization, by default None.
        **kwargs : Any
            Additional keyword arguments passed to `MDYPLLogisticResult`.

        Returns
        -------
        MDYPLLogisticResult
            Fitted results instance containing parameter estimates, diagnostics,
            and methods for high-dimensional inference.
        """
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
        """Formula-based construction is not supported.

        Raises
        ------
        NotImplementedError
            Always raised. Specify design matrix `exog` and response `endog`
            directly when instantiating `MDYPLLogistic`.
        """
        raise NotImplementedError(
            "Formula-based model specification via `from_formula` is not supported "
            "for `MDYPLLogistic`. Pass explicit `endog` and `exog` design matrices"
            " directly."
        )
