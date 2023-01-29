from __future__ import annotations

import numpy as np
from typing import List
from quantus.helpers.model.model_interface import ModelInterface


class BatchedRobustnessMetric:

    """Common functionality for batched robustness metrics."""

    return_nan_when_prediction_changes: bool

    def changed_prediction_indices(
        self, model: ModelInterface, x_batch: np.ndarray, x_perturbed: np.ndarray
    ) -> np.ndarray | List:
        """Check if applying perturbation caused models predictions to change."""
        if not self.return_nan_when_prediction_changes:
            return []

        labels_before = model.predict(x_batch).argmax(axis=-1)
        labels_after = model.predict(x_perturbed).argmax(axis=-1)
        return np.argwhere(labels_before != labels_after).reshape(-1)
