from functools import cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermite


@cache
def get_hermite_roots_weights(
    n: int = 200,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes and caches the roots and weights of a Hermite polynomial to be
    used in Gauss-Hermite quadrature to approximate an integral.

    Parameters
    ----------
    n : int, optional
        Number of nodes to be used in Gauss Hermite quadrature , by default 200.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Tuple of nodes and corresponding weights to approximate integrals.
    """
    return roots_hermite(n)
