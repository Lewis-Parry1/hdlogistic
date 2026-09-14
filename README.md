# hdlogistic

[**hdlogistic**](https://github.com/Lewis-Parry1/hdlogistic) provides a Python implementation of R's [**brglm2**](https://github.com/ikosmidis/brglm2/) for estimating parameters of a logistic regression model using the Diaconis–Ylvisaker (DY) penalised likelihood [1] and provides the rescaled estimates in finite sample settings where the ratio of covariates to observations is large. It also offers subsequent inference procedures and tools for model comparison, both of which can be adjusted to account for the high dimensional regime. 

In the context of logistic regression, the DY prior penalised likelihood yields an asymptotically smaller bias than the standard maximum likelihood estimator (MLE), guarantees finite parameter estimates [2], and is computationally more efficient than other non-separable penalties, such as Jeffrey' invariant prior penalty [2] For maximum DY prior penalised likelihood (MDYPL) estimation, the package provides a non-linear state equation solver for the axillary equations derived from the Approximate Message Passing (AMP) recursion constructed in Sterzinger & Kosmids [3]. 

In the high-dimensional setting where $p,n\to\infty$ such that $p/n\to\kappa\in (0,1)$, the MDYPL estimate, like the standard MLE, is asymptotically biased, exhibits inflated standard errors, and the likelihood ratio test (LRT) statistic converges to a rescaled chi-squared distribution. By solving the auxiliary system of equations from Sterzinger & Kosmidis [3], this package compute rescaled, asymptotically unbiased MDYPL estimates, adjusted standard errors, and valid confidence intervals, alongside a rescaled penalised LRT.


## Quick start

After cloning the repository, create and activate the project environment, then install the package in editable mode:

```bash
uv sync
uv run python -c "import hdlogistic; print(hdlogistic.__file__)"
```

This ensures the package is installed from the local source tree and ready to import.

## Documentation

To build the Sphinx documentation locally, run `uv run sphinx-build -b html docs/source docs/build/html`, then open `docs/build/html/index.html` in a browser to view it.

## CI

This repository includes a GitHub Actions workflow at `.github/workflows/pre-commit.yml` that runs `pre-commit run --all-files` on pushes and pull requests.

## References and Resources

[1] Rigon, T. and Aliverti, E. (2023). Conjugate priors and bias reduction for logistic regression models. *Statistics & Probability Letters*, 202, 3. https://doi.org/10.1016/j.spl.2023.109901
(See p. 3 for existence and uniqueness of the Diaconis–Ylvisaker (DY) estimator.)

[2] Firth, D. (1992). Bias reduction, the Jeffreys prior and GLIM. In L. Fahrmeir, B. Francis, R. Gilchrist, & G. Tutz (Eds.), *Advances in GLIM and Statistical Modelling* (pp. 91–100). Springer New York.

[3] Sterzinger, P. and Kosmidis, I. (2026). Diaconis–Ylvisaker prior penalized likelihood for $p/n \to \kappa \in (0,1)$ logistic regression. *Biometrika*, 113(2), 1–23. https://doi.org/10.1093/biomet/asag014

