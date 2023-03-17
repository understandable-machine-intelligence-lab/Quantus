# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import ExplainFn, Explanation
from quantus.nlp.helpers.utils import (
    get_input_ids,
    is_plain_text_perturbation,
    value_or_default,
)
from quantus.nlp.metrics.robustness.internal.batched_perturbation_metric import (
    BatchedPerturbationMetric,
)


class RobustnessMetric(BatchedPerturbationMetric, ABC):
    def __init__(self, nr_samples: int, **kwargs):
        super().__init__(**kwargs)
        self.nr_samples = nr_samples

    def _batch_preprocess(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
    ) -> Tuple[List[str], np.ndarray, List[Explanation], Optional[None]]:
        if not is_plain_text_perturbation(self.perturb_func):
            return super()._batch_preprocess(
                model, x_batch, y_batch, a_batch, explain_func, explain_func_kwargs
            )

        y_batch = value_or_default(
            y_batch, lambda: model.predict(x_batch).argmax(axis=-1)
        )

        batch_size = len(x_batch)
        # For plain text we need to first collect perturbations, then
        # "pre-tokenize" them, so we end up with sequences all the same length.
        x_perturbed_batches = [x_batch] + [
            self.perturb_func(x_batch, **self.perturb_func_kwargs)  # noqa
            for _ in range(self.nr_samples)
        ]
        x_perturbed_batches = np.reshape(x_perturbed_batches, -1).tolist()
        x_perturbed_ids, _ = get_input_ids(x_perturbed_batches, model)  # noqa
        x_perturbed_batches = model.batch_decode(x_perturbed_ids)
        x_batch, x_perturbed_batches = (
            x_perturbed_batches[:batch_size],
            x_perturbed_batches[batch_size:],
        )
        x_perturbed_batches = np.reshape(x_perturbed_batches, (self.nr_samples, -1))

        # We need to pre-compute explanations for padded x_batch
        model.batch_encode = partial(model.batch_encode, add_special_tokens=False)  # type: ignore
        a_batch = self._explain_batch(
            model, x_batch, y_batch, explain_func, explain_func_kwargs
        )

        return x_batch, y_batch, a_batch, x_perturbed_batches.tolist()  # type: ignore

    def _batch_postprocess(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
        score: np.ndarray,
    ) -> np.ndarray:
        if is_plain_text_perturbation(self.perturb_func):
            model.batch_encode = model.batch_encode.func  # type: ignore
        return super()._batch_postprocess(
            model, x_batch, y_batch, a_batch, explain_func, explain_func_kwargs, score
        )

    def _evaluate_batch(  # type: ignore
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
        custom_batch: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        batch_size = len(x_batch)
        scores = np.full((self.nr_samples, batch_size), fill_value=np.NINF)

        for step_id in range(self.nr_samples):
            if is_plain_text_perturbation(self.perturb_func):
                scores[step_id] = self._evaluate_step_plain_text_noise(
                    model,
                    x_batch,
                    y_batch,
                    a_batch,
                    explain_func,
                    explain_func_kwargs,
                    x_perturbed=custom_batch[step_id],
                )
            else:
                scores[step_id] = self._evaluate_step_latent_space_noise(
                    model,
                    x_batch,
                    y_batch,
                    a_batch,
                    explain_func,
                    explain_func_kwargs,
                )
        return self._aggregate_instances(scores)

    @abstractmethod
    def _evaluate_step_latent_space_noise(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _evaluate_step_plain_text_noise(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
        x_perturbed: List[str],
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _aggregate_instances(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: not covered
