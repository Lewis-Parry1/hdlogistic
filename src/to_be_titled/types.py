from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

if TYPE_CHECKING:
    from to_be_titled.mdypl_fit import MDYPLModel
    from to_be_titled.penalised_likelihood_ratio_test import PenalisedLRTResults

type FloatArray = NDArray[np.float64]


class MDYPLResults:
    def __init__(
        self,
        model: MDYPLModel,
        glm_results: GLMResultsWrapper | Any,
        skip_null_deviance: bool = False,
    ) -> None:
        self.model = model
        self._results = glm_results

        self.y_raw = model.y_raw
        self.y_adj = model.y_adj
        self.alpha = model.alpha

        self.has_intercept = model.has_intercept
        self.intercept_idx = model.intercept_idx
        self.prior_weights = model.prior_weights

        self.iterations = glm_results.fit_history.get("iteration")

        self.params = np.asarray(glm_results.params, dtype=np.float64)
        self.fitted_probs = np.asarray(self.fittedvalues, dtype=np.float64).copy()
        self.linear_predictors = (
            cast(FloatArray, model.exog) @ self.params + model.offset
        )

        self.residuals = (model.y_raw - self.fitted_probs) / (
            self.fitted_probs * (1.0 - self.fitted_probs)
        )
        # TODO: This needs to be change to determiend matrix rank
        # If matrix is not full rank then this is incorrect
        self.rank = len(self.params)

        from to_be_titled.inference import logist_aic  # TODO: Get rid of crap like this

        self.aic = (
            logist_aic(self.y_adj, self.fitted_probs, self.prior_weights)
            + 2.0 * self.rank
        )
        self.deviance = glm_results.deviance
        # TODO: skip_null_deviance stops us from getting stuck in loop
        # if we didnt have it we would fit full model -> fit null model
        # fit null model again and again...
        # I think we could fix this by seperating _compute_null_deviance
        self.null_deviance = (
            float("nan") if skip_null_deviance else self._compute_null_deviance(model)
        )

    def __getattr__(self, name: str) -> Any:
        # Handy lookup
        return getattr(self._results, name)

    @property
    def intercept(self) -> float | None:
        """The fitted intercept coeffcient, or None if the model has
        no intercept.
        """
        if self.intercept_idx is None:
            return None
        return float(self.params[self.intercept_idx])

    @cached_property
    def leverages(self) -> FloatArray:
        """The diaganal of the hat matrix. Cached as to not need to recompute."""
        return np.asarray(self._results.get_hat_matrix_diag(), dtype=np.float64)

    def _compute_null_deviance(self, model: MDYPLModel) -> float:
        """Compute deviance of model fitted with only constant feature."""
        x = cast(FloatArray, model.exog)
        intercept_idx = cast(int, model.intercept_idx)

        family = model.family

        if model.has_intercept:
            from to_be_titled.mdypl_fit import MDYPLModel  # TODO: Get rid of this

            intercept_col = x[:, intercept_idx : intercept_idx + 1]

            # This just gets an initial starting estimate
            # for intercept (average of y) and applied
            y_mean = float(np.mean(self.y_raw))
            start_val = model.family.link(y_mean)  # logit(y_bar)
            start_params = np.asarray([start_val])

            # Recursive -> call MDYPL fit full model
            # Create MDYPLResults to save full model results
            # MDYPLResults built also calls MDYPL fit in order
            # To fit intercept only model
            null_model = MDYPLModel(
                y=self.y_raw,
                x=intercept_col,
                weights=self.prior_weights,
                offset=model.offset,  # these will just be zeroes
                alpha=self.alpha,
                family=model.family,
                fit_kwargs=model.fit_kwargs,
            )
            # Fit null model with smart start.
            # get fitted probs
            null_mus = null_model.fit(
                start_params=start_params, skip_null_deviance=True
            ).fitted_probs

            # Compute deviance using usual deviance formula
            # Given y_adj, null_mus and weights
            return float(family.deviance(self.y_adj, null_mus, self.prior_weights))

        else:
            # Null model now has no intercept, now either
            # 1. If has offset -> its the sigmoid(offset)
            # 2. If no offset then its sigmoid(0) = 1/2 for all
            null_mus = model.family.link.inverse(model.offset)
            return float(family.deviance(self.y_adj, null_mus, self.prior_weights))

    def penalised_lrt(
        self,
        other: MDYPLResults,
        hd_correction: bool = False,
        solve_se_kwargs: dict[str, Any] | None = None,
    ) -> PenalisedLRTResults:
        """Penalized likelihood ratio test against a nested MDYPL fit.

        `self` and `other` may be either the full or restricted model —
        order does not matter.
        """
        from to_be_titled.penalised_likelihood_ratio_test import penalised_lrt

        return penalised_lrt(
            self, other, hd_correction=hd_correction, solve_se_kwargs=solve_se_kwargs
        )
