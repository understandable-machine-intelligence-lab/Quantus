from __future__ import annotations

from abc import ABC

import numpy as np
from typing import List, Optional

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.metrics.batched_perturbation_metric import BatchedPerturbationMetric


class RobustnessMetric(BatchedPerturbationMetric, ABC):

    """Common functionality for batched robustness metrics."""

    def __init__(self, *args, return_nan_when_prediction_changes, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_nan_when_prediction_changes = return_nan_when_prediction_changes

    def indexes_of_changed_predictions_plain_text(
        self, model: TextClassifier, x_batch: List[str], x_batch_perturbed: List[str]
    ) -> np.ndarray | List:
        """Check if applying perturbation caused models predictions to change using plain text."""
        if not self.return_nan_when_prediction_changes:
            return []

        labels_before = model.predict(x_batch).argmax(axis=-1)
        labels_after = model.predict(x_batch_perturbed).argmax(axis=-1)
        return np.argwhere(labels_before != labels_after).reshape(-1)

    def indexes_of_changed_predictions_latent(
        self,
        model: TextClassifier,
        x_batch_embeddings: np.ndarray,
        x_batch_perturbed: np.ndarray,
        **kwargs,
    ) -> np.ndarray | List:
        """Check if applying perturbation caused models predictions to change using latent representations."""
        if not self.return_nan_when_prediction_changes:
            return []

        labels_before = model(x_batch_embeddings, **kwargs).argmax(axis=-1)
        labels_after = model(x_batch_perturbed, **kwargs).argmax(axis=-1)
        return np.argwhere(labels_before != labels_after).reshape(-1)
