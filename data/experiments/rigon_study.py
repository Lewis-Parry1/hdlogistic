from pathlib import Path

import numpy as np
from scipy.special import expit

from to_be_titled.estimation import fit_diaconis_ylvisaker_logistic_regression
from to_be_titled.summary import summary


def aggregate_bias(estimates: np.ndarray, true_betas: np.ndarray) -> float:
    return float(np.mean(estimates - true_betas))


script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data"

X = np.loadtxt(data_dir / "X_data.csv", delimiter=",", skiprows=1)
y = np.loadtxt(data_dir / "y_data.csv", delimiter=",", skiprows=1)
betas_true = np.loadtxt(data_dir / "betas_true.csv", delimiter=",")

n, p = X.shape
print(f"Loaded dataset: n={n}, p={p}")

probabilities = expit(X @ betas_true)
y = np.random.binomial(n=1, p=probabilities)

alpha_default = 2 / 3

result = fit_diaconis_ylvisaker_logistic_regression(
    x=X,
    y=y,
    alpha=alpha_default,
    intercept_index=None,
    solver="fisher_scoring",
)

raw_betas = summary(result, high_dimensional_correction=False)
corrected_betas = summary(result, high_dimensional_correction=True)

raw_bias = aggregate_bias(raw_betas, betas_true)
corrected_bias = aggregate_bias(corrected_betas, betas_true)

print(f"Default Alpha (n / (n + p)): {alpha_default:.4f}")
print(f"Aggregate Bias (Standard MDYPL): {raw_bias:.6f}")
print(f"Aggregate Bias (Rescaled/HD Corrected): {corrected_bias:.6f}")