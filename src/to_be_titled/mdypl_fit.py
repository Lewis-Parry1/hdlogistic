import numpy as np
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit


from to_be_titled.types import FloatArray, MDYPLResults
from to_be_titled.utils import _adjust_response, _has_constant_col, _get_intercept_idx
from to_be_titled.validation import _ensure_column_vector, _ensure_design_matrix


class MDYPLModel(GLM):
    """MDYPL logistic regression. Fits a Binomial GLM on pseudo-responses,
    following Sterzinger & Kosmidis (2026). 

    Parameters
    ----------
    y : FloatArray
        Raw binary response variables in {0,1}. Array should have shape (n,), where `n`
        is the number of observations. 
    x: FloatArray 
        Design matrix. Include a constant column (e.g via `sm.add_constant`) 
        if an intercept is wanted. 
    alpha: float | None
        Shrinkage parameter in [0,1]. Defaults to 
        `sum(weights) / (sum(weights) + wp - has_intercept)`. 
    weights: FloatArray | None 
        Prior / frequency weights, passed through to `GLM` as `freq_weights`.
        By default, None. 
    offset: float | FloatArray | None
        Offset to include in the linear predictor. By default, None. 
    family: statsmodels Family 
        Currently only `Binomial()` with the logit link is supported. 
    fit_kwargs : dict | None
        Accepted fit arguments are `tol`, `maxiter`, `method` and `start_params` 
        to be passed into `.fit()`. 
    """
    def __init__(self,
                 y: FloatArray, 
                 x: FloatArray, 
                 alpha: float | None = None,
                 weights : FloatArray | None = None,
                 offset: float | FloatArray | None = None, 
                 family = Binomial(),
                 fit_kwargs: dict | None = None,
                 ):
        
        # Shape checks for design matrix and response 
        x_val = _ensure_design_matrix(x)
        y_val = _ensure_column_vector(y)

        # TODO: Full rank check? 

        n, p = x_val.shape[0], x_val.shape[1]
        has_intercept = _has_constant_col(x_val)
        intercept_idx = _get_intercept_idx(x_val)

        if not (isinstance(family, Binomial) and isinstance(family.link, Logit)):
            raise ValueError("MDYPLModel currently only supports Binomial family" \
            "with logit link.")

        if weights is None: 
            weights = np.ones(n)

        if offset is None: 
            offset = np.zeros(n)

        if alpha is None: 
            wsum = float(np.sum(weights))
            alpha = wsum / (wsum + p -  int(has_intercept))
        elif not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Shrinkage paramater `alpha` must be in [0,1], got {alpha}")

        # Transform binary responses to MDYPL pseudo responses 
        # Assumes prior mode is the zero vector 
        y_adj = _adjust_response(y_val, alpha)

        # Build `statsmodels.GLM` 
        super().__init__(
            endog=y_adj,
            exog=x_val,
            family=family,
            freq_weights=weights,
            offset=offset,
        )

        if bool(self.k_constant) != has_intercept:
            raise ValueError("Internal mismatch between detected intercept in design" \
            "matrix and `statsmodels` constant column detection.")

        self.y_raw = np.asarray(y_val, dtype=np.float64)
        self.y_adj = np.asarray(y_adj, dtype= np.float64)
        self.alpha = alpha 
        self.offset = offset
        self.has_intercept = has_intercept
        self.intercept_idx = intercept_idx
        self.fit_kwargs = fit_kwargs or {}
    
    def fit(self, **kwargs) -> MDYPLResults:
        fit_kwargs = {**self.fit_kwargs, **kwargs}
        allowed_args = {"tol", "maxiter", "method", "start_params"}
        unexpected = set(fit_kwargs) - allowed_args
        if unexpected: 
            raise ValueError(f"Unexpected fit arguement(s): {sorted(unexpected)}")

        glm_results = super().fit(**fit_kwargs)

        return MDYPLResults(self, glm_results)
        
