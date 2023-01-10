from __future__ import annotations

import numpy as np
from quantus.nlp.metrics.robustness.sensitivity import Sensitivity


class MaxSensitivity(Sensitivity):
    def aggregate_steps(self, arr: np.ndarray) -> float | np.ndarray:
        agg_fn = np.max if self.return_nan_when_prediction_changes else np.nanmax
        return agg_fn(arr, axis=1)
