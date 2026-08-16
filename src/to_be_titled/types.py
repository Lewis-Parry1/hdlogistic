from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from to_be_titled.mdypl_fit import MDYPLModel
from to_be_titled.summary import 

type FloatArray = NDArray[np.float64]

class MDYPLResults:
    def __init__(self, model: MDYPLModel, glm_results):
        self.model = model
        self._results = glm_results

        self.y_raw = model.y_raw
        self.y_adj = model.y_adj
        self.alpha = model.alpha 
        self.has_intercept = model.has_intercept
        self.intercept_idx = model.intercept_idx

    def __getattr__(self, name): 
        return getattr(self._results, name)

    @property
    def intercept(self) -> float | None: 
        """The fitted intercept coeffcient, or None if the model has 
        no intercept. 

        Returns
        -------
        float | None
            Intercept of the fitted MDYPL model. 
        """

