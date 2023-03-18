# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import ABC
from typing import List
import numpy as np

from quantus.helpers.utils import expand_attribution_channel
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.base_batched import BatchedPerturbationMetric


class BatchedRobustnessMetric(BatchedPerturbationMetric, ABC):
    def __init__(self, *args, return_nan_when_prediction_changes: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_nan_when_prediction_changes = return_nan_when_prediction_changes

    def changed_prediction_indices(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        x_perturbed: np.ndarray,
    ) -> List[int]:
        """Get indices of inputs, for which prediction chnaged."""
        if not self.return_nan_when_prediction_changes:
            return []

        predicted_y = model.predict(x_batch).argmax(axis=-1)
        predicted_y_perturbed = model.predict(x_perturbed).argmax(axis=-1)

        changed_prediction_indices = np.argwhere(predicted_y != predicted_y_perturbed)
        return np.reshape(changed_prediction_indices, -1).tolist()  # noqa

    def generate_normalised_explanations_batch(
        self, model: ModelInterface, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> np.ndarray:
        """
        Generate explanation, apply normalization and take absolute values if configured so during metric instantiation.

        Parameters
        ----------
        model:
            Model which is subject tpo explanation.
        x_batch: np.ndarray
            4D tensor representing batch of input images.
        y_batch: np.ndarray
             1D tensor, representing predicted labels for the x_batch.

        Returns
        -------
        a_batch: np.ndarray
            A batch of explanations.
        """
        a_batch = self.explain_func(
            inputs=x_batch,
            targets=y_batch,
            model=model.get_model(),
            **self.explain_func_kwargs
        )
        if self.normalise:
            a_batch = self.normalise_func(a_batch, **self.normalise_func_kwargs)
        if self.abs:
            a_batch = np.abs(a_batch)
        return expand_attribution_channel(a_batch, x_batch)
