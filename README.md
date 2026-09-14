# hdlogistic

[**hdlogistic**](https://github.com/Lewis-Parry1/hdlogistic) provides a Python implementation of R's [**brglm2**](https://github.com/ikosmidis/brglm2/) for estimating parameters of a logistic regression model using the Diaconis–Ylvisaker (DY) penalised likelihood [1] and provides the rescaled estimates in finite sample settings where the ratio of covariates to observations is large. It also offers subsequent inference procedures and tools for model comparison, both of which can be adjusted to account for a high dimensional regime.

In the context of logistic regression, the DY prior penalised likelihood yields an asymptotically smaller bias than the standard maximum likelihood estimator (MLE), guarantees finite parameter estimates [2, 4], and is computationally more efficient than other non-separable penalties, such as Jeffreys' invariant prior penalty [4]. For maximum DY prior penalised likelihood (MDYPL) estimation, the package provides a non-linear state equation solver for the auxiliary equations derived from the Approximate Message Passing (AMP) recursion constructed in Sterzinger & Kosmidis [3].

In the high-dimensional setting where $p,n\to\infty$ such that $p/n\to\kappa\in (0,1)$, the MDYPL estimate, like the standard MLE, is asymptotically biased, exhibits inflated standard errors, and the likelihood ratio test (LRT) statistic converges to a rescaled chi-squared distribution. By solving the auxiliary system of equations from Sterzinger & Kosmidis [3], this package computes rescaled, asymptotically unbiased MDYPL estimates, adjusted standard errors, and valid confidence intervals, alongside a rescaled penalised LRT.

## Quick start

After cloning the repository, create and activate the project environment, then install the package in editable mode:

```bash
uv sync
uv run python -c "import hdlogistic; print(hdlogistic.__file__)"
```

This ensures the package is installed from the local source tree and ready to import.

## Documentation

To build the Sphinx documentation locally, run `uv run sphinx-build -b html docs/source docs/build/html`, then open `docs/build/html/index.html` in a browser to view it.

## Example Usage 

Consider the [Multiple Features dataset](https://archive.ics.uci.edu/dataset/72/multiple+features) containing digits (0-9) obtained from a collection of Dutch public utility maps. It contains 200 different 30 x 48 handwritten variations of each digit from 0-9. This example is inspired by that in [**brglm2**](https://github.com/ikosmidis/brglm2/) and extracts the columns in the dataset representing the Fourier coefficients and Karhunen-Loeve coefficients.

```python
import pandas as pd
import numpy as np

from hdlogistic import MDYPLLogistic

CSV_PATH = # YOUR FILE PATH GOES HERE

df = pd.read_csv(CSV_PATH)

vars_ = [c for c in df.columns if c.startswith('fou') or c.startswith('kar')]
nest_vars = [c for c in vars_ if c.startswith('fou')]
```

The response vector (digit from 0-9) and the corresponding design matrices for the fitted and nested model are:
```python
# get response
y = np.asarray((df['digit'] == 7).astype(float).values)

# full model features (fou + kar)
X_full = df[vars_].values.astype(float)

# nested model features (fou only)
X_nest = df[nest_vars].values.astype(float)
```

The full model is fit, and the nested model used in the penalised likelihood ratio test is also fitted with the same shrinkage parameter, alpha, used to transform the responses that the full model was fit on.
```python
# compute the mdypl fits
full_model = MDYPLLogistic(endog=y, exog=X_full)
full_fit = full_model.fit()

# extract the default shrinkage parameter used in the full model
alpha = full_fit.alpha

# shrinkage parameter used in full model must also be used in nested model
nested_model = MDYPLLogistic(endog=y, exog=X_nest, alpha=alpha)
nested_fit = nested_model.fit()
```

To apply the high dimensional corrections to the estimates, standard errors, and confidence intervals, and print results:
```python
# get high dimensional corrections
full_hd_corrected = full_fit.get_high_dimensional_result()

# Get full model summary
summ_hd = full_hd_corrected.summary()
print(summ_hd)
```

The penalised likelihood ratio test with the high dimensional correction applied is conducted and printed as follows:
```python
# conduct penalised likelihood ratio test with high dim correction applied
plr_results = full_fit.penalised_lrt(nested_fit, hd_correction=True)
print(plr_results)
```

## CI

This repository includes a GitHub Actions workflow at `.github/workflows/pre-commit.yml` that runs `pre-commit run --all-files` on pushes and pull requests.

## References and Resources

[1] Rigon, T. and Aliverti, E. (2023). Conjugate priors and bias reduction for logistic regression models. *Statistics & Probability Letters*, 202, 3. https://doi.org/10.1016/j.spl.2023.109901
(See p. 3 for existence and uniqueness of the Diaconis–Ylvisaker (DY) estimator.)

[2] Firth, D. (1992). Bias reduction, the Jeffreys prior and GLIM. In L. Fahrmeir, B. Francis, R. Gilchrist, & G. Tutz (Eds.), *Advances in GLIM and Statistical Modelling* (pp. 91–100). Springer New York.

[3] Sterzinger, P. and Kosmidis, I. (2026). Diaconis–Ylvisaker prior penalized likelihood for $p/n \to \kappa \in (0,1)$ logistic regression. *Biometrika*, 113(2), 1–23. https://doi.org/10.1093/biomet/asag014

[4] Kosmidis, I. and Firth, D. (2021). Jeffreys-prior penalty, finiteness and shrinkage in binomial-response generalized linear models. *Biometrika*, 108, 71–82. https://doi.org/10.1093/biomet/asaa052
