# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import Optional, Callable, Dict, List
import numpy as np


from quantus.nlp.helpers.types import (
    PerturbationType,
    NumericalPerturbFn,
    PlainTextPerturbFn,
    NormaliseFn,
    TextClassifier,
    Explanation
)
from quantus.nlp.metrics.batched_text_classification_metric import (
    BatchedTextClassificationMetric,
)
from quantus.nlp.metrics.robustness.batched_robustness_metric import (
    BatchedRobustnessMetric,
)
from quantus.helpers.asserts import attributes_check
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.functions.perturb_func import spelling_replacement
from quantus.nlp.helpers.utils import unpack_token_ids_and_attention_mask, safe_asarray
from quantus.metrics.robustness.relative_stability.ris_objective import RelativeInputStabilityObjective


class RelativeInputStability(BatchedTextClassificationMetric, BatchedRobustnessMetric):
    """
    Relative Input Stability leverages the stability of an explanation with respect to the change in the input data

    :math:`RIS(x, x', e_x, e_x') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{x - x'}{x}||_p, \epsilon_{min})}`

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/abs/2203.06877
    """

    @attributes_check
    def __init__(
        self,
        *,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: NormaliseFn = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        perturbation_type: PerturbationType = PerturbationType.plain_text,
        perturb_func: PlainTextPerturbFn | NumericalPerturbFn = spelling_replacement,
        perturb_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable[[np.ndarray], np.float] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        eps_min: float = 1e-6,
        default_plot_func: Optional[Callable] = None,
        return_nan_when_prediction_changes: bool = True,
        **kwargs,
    ):
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturbation_type=perturbation_type,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        self.nr_samples = nr_samples
        self.objective = RelativeInputStabilityObjective(eps_min)

    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        **kwargs,
    ) -> np.ndarray | float:

        batch_size = len(x_batch)

        tokenized_input = model.tokenizer.tokenize(x_batch)
        input_ids, attention_mask = unpack_token_ids_and_attention_mask(tokenized_input)
        x_batch_embeddings = safe_asarray(model.embedding_lookup(input_ids))

        ris_batch = np.zeros(shape=[self.nr_samples, batch_size])

        for step_id in range(self.nr_samples):
            if self.perturbation_type == PerturbationType.plain_text:
                ris_batch[step_id] = self._evaluate_batch_step_plain_text_noise(
                    model, x_batch, y_batch, a_batch, x_batch_embeddings
                )
            if self.perturbation_type == PerturbationType.latent_space:
                a_batch_numerical = np.asarray([i[1] for i in a_batch])
                ris_batch[step_id] = self._evaluate_batch_step_latent_space_noise(
                    model,
                    y_batch,
                    a_batch_numerical,
                    x_batch_embeddings,
                    attention_mask,
                )

        # Compute RIS.
        result = np.max(ris_batch, axis=0)
        if self.return_aggregate:
            result = [self.aggregate_func(result)]
        return result

    def _evaluate_batch_step_plain_text_noise(
            self,
            model: TextClassifier,
            x_batch: List[str],
            y_batch: np.ndarray,
            a_batch: List[Explanation],
            x_batch_embeddings: np.ndarray,
    ) -> np.ndarray | float:
        pass

    def _evaluate_batch_step_latent_space_noise(
            self,
            model: TextClassifier,
            y_batch: np.ndarray,
            a_batch: np.ndarray,
            x_batch_embeddings: np.ndarray,
            attention_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        # Perturb input.
        x_perturbed = self.perturb_func(x_batch_embeddings, **self.perturb_func_kwargs)
        # Generate explanations for perturbed input.
        a_batch_perturbed = self.explain_func(
            model, x_perturbed, y_batch, attention_mask, **self.explain_func_kwargs
        )
        # Compute maximization's objective.
        ris = self.objective(
            x_batch_embeddings, x_perturbed, a_batch, a_batch_perturbed
        )
        # We're done with this sample if `return_nan_when_prediction_changes`==False.
        if not self.return_nan_when_prediction_changes:
            return ris

        # If perturbed input caused change in prediction, then it's RIS=nan.
        changed_prediction_indices = self.indexes_of_changed_predictions_latent(
            model, x_batch_embeddings, x_perturbed, attention_mask
        )

        if len(changed_prediction_indices) != 0:
            ris[changed_prediction_indices] = np.nan
