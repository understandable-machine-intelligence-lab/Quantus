# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


from quantus.nlp.helpers.types import ExplainFn, Explanation
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import get_input_ids, get_scores, safe_as_array
from quantus.nlp.metrics.robustness.internal.robustness_metric import RobustnessMetric


class RelativeStability(RobustnessMetric, ABC):
    """
    Relative Input Stability leverages the stability of an explanation with respect to the change in the input data

    :math:`RIS(x, x', e_x, e_x') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{x - x'}{x}||_p, \epsilon_{min})}`

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/abs/2203.06877
    """

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
        x_embeddings = model.embedding_lookup(input_ids)
        x_embeddings = safe_as_array(x_embeddings)
        # fmt: off
        x_perturbed = self.perturb_func(x_embeddings, **self.perturb_func_kwargs)  # noqa
        a_batch_perturbed = self._explain_batch(model, x_perturbed, y_batch, explain_func, explain_func_kwargs)
        # fmt: on
        a_batch_scores = get_scores(a_batch)
        # Compute maximization's objective.
        rs = self._compute_objective_latent_space(
            model,
            x_embeddings,
            x_perturbed,
            a_batch_scores,
            a_batch_perturbed,
            predict_kwargs,
        )
        return rs

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
        """Latent-space perturbation."""
        # Generate explanations for perturbed input.
        a_batch_perturbed = self._explain_batch(
            model, x_perturbed, y_batch, explain_func, explain_func_kwargs
        )

        # Compute maximization's objective.
        rs = self._compute_objective_plain_text(
            model,
            x_batch,
            x_perturbed,
            a_batch,
            a_batch_perturbed,
        )
        return rs

    def _aggregate_instances(self, scores: np.ndarray) -> np.ndarray:
        return np.nanmax(scores, axis=0)

    @abstractmethod
    def _compute_objective_plain_text(
        self,
        model: TextClassifier,
        x_batch: List[str],
        x_batch_perturbed: List[str],
        a_batch: List[Explanation],
        a_batch_perturbed: List[Explanation],
    ) -> np.ndarray:
        raise NotImplementedError  # pragma: not covered

    @abstractmethod
    def _compute_objective_latent_space(
        self,
        model: TextClassifier,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        predict_kwargs: Dict,
    ) -> np.ndarray:
        raise NotImplementedError  # pragma: not covered
