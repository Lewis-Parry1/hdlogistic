from functools import cache

from scipy.special import roots_hermite

from to_be_titled._types import FloatArray


@cache
def get_hermite_roots_weights(
    n: int = 200,
) -> tuple[FloatArray, FloatArray]:
    """
    Computes and caches the roots and weights of a Hermite polynomial to be
    used in Gauss-Hermite quadrature to approximate an integral.

    Parameters
    ----------
    n : int, optional
        Number of nodes to be used in Gauss Hermite quadrature , by default 200.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Tuple of nodes and corresponding weights to approximate integrals.
    """
    return roots_hermite(n)
