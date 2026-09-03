import numpy as np
from numpy.typing import NDArray
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

if TYPE_CHECKING:
    from to_be_titled.mdypl_fit import MDYPLModel
    from to_be_titled.penalised_likelihood_ratio_test import PenalisedLRTResults

type FloatArray = NDArray[np.float64]
