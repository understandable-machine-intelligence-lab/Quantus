# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import ABC
from typing import Dict, List

import numpy as np

from quantus.functions.similarity_func import difference
from quantus.helpers.warn import warn_perturbation_caused_no_change
from quantus.nlp.helpers.types import (
    ExplainFn,
    Explanation,
    NormFn,
    SimilarityFn,
    TextClassifier,
)
from quantus.nlp.helpers.utils import get_input_ids, get_scores, safe_as_array
from quantus.nlp.metrics.robustness.internal.robustness_metric import RobustnessMetric


class SensitivityMetric(RobustnessMetric, ABC):
    """Shared implementation for Avg Sensitivity, Max Sesnitivity."""

    def __init__(
        self,
        *args,
        norm_numerator: NormFn,
        norm_denominator: NormFn,
        similarity_func: SimilarityFn = difference,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Save metric-specific attributes.
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator
        self.similarity_func = similarity_func

    def _evaluate_step_latent_space_noise(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
    ) -> np.ndarray:
        input_ids, predict_kwargs = get_input_ids(x_batch, model)
        x_embeddings = safe_as_array(model.embedding_lookup(input_ids))
        # fmt: off
        x_perturbed = self.perturb_func(x_embeddings, **self.perturb_func_kwargs)  # noqa
        # fmt: on

        warn_perturbation_caused_no_change(x_embeddings, x_perturbed)

        a_batch_perturbed = self._explain_batch(
            model,
            x_perturbed,
            y_batch,
            explain_func,
            {**explain_func_kwargs, **predict_kwargs},
        )
        a_batch_scores = get_scores(a_batch)

        sensitivities = self.similarity_func(
            a_batch_scores,
            a_batch_perturbed,
        )

        numerator = self.norm_numerator(sensitivities)
        denominator = self.norm_denominator(x_embeddings)

        similarities = numerator / denominator

        return similarities

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
        warn_perturbation_caused_no_change(x_batch, x_perturbed)  # noqa

        a_batch_perturbed = self._explain_batch(
            model,
            x_perturbed,
            y_batch,
            explain_func,
            explain_func_kwargs,
        )

        a_batch_scores = get_scores(a_batch)
        a_batch_perturbed_scores = get_scores(a_batch_perturbed)

        sensitivities = self.similarity_func(
            a_batch_scores,
            a_batch_perturbed_scores,
        )

        x_ids, _ = get_input_ids(x_batch, model)
        x_embeddings = model.embedding_lookup(x_ids)

        numerator = self.norm_numerator(sensitivities)
        denominator = self.norm_denominator(safe_as_array(x_embeddings))

        similarities = numerator / denominator

        return similarities
