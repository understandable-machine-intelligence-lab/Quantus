from __future__ import annotations

import numpy as np
from quantus.nlp.metrics.robustness.sensitivity import Sensitivity


class AvgSensitivity(Sensitivity):
    def aggregate_steps(self, arr: np.ndarray) -> float | np.ndarray:
        agg_fn = np.mean if self.return_nan_when_prediction_changes else np.nanmean
        return agg_fn(arr, axis=1)
