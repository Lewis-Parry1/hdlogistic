"""
run_summary_check.py

Standalone integration script: fits the MDYPL model on the fixed 10-row
reference dataset and prints both the standard and HD-corrected summaries,
for comparison against the equivalent R script.

Run with:
    python run_summary_check.py
"""

import numpy as np

from to_be_titled.adapters.statsmodels import MDYPLLogistic


def main() -> None:
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

    # Full model: intercept + both covariates
    full_model = MDYPLLogistic(endog=y, exog=x)
    full_result = full_model.fit()

    print(full_result.summary())

    print()

    hd_result = full_result.get_high_dimensional()
    print(hd_result.summary())


if __name__ == "__main__":
    main()