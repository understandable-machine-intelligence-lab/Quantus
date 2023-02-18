from __future__ import annotations

from quantus.nlp.metrics.robustness.internal.robustness_metric import RobustnessMetric
from abc import abstractmethod
import numpy as np
from typing import List, Callable, Optional


from quantus.functions.similarity_func import difference
from quantus.nlp.helpers.types import (
    Explanation,
    TextClassifier,
    PerturbationType,
    SimilarityFn,
)
from quantus.nlp.helpers.utils import (
    unpack_token_ids_and_attention_mask,
    safe_asarray,
    explanation_similarity,
)
from quantus.helpers.warn import warn_perturbation_caused_no_change


class SensitivityMetric(RobustnessMetric):
    """Shared implementation for Avg Sensitivity, Max Sesnitivity, Local Lipschitz Estimate"""

    def __init__(
        self,
        *args,
        norm_numerator: Callable[[np.ndarray], float],
        norm_denominator: Callable[[np.ndarray], float],
        nr_samples: int,
        similarity_func: SimilarityFn = difference,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Save metric-specific attributes.
        self.nr_samples = nr_samples
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator
        self.similarity_func = similarity_func

    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation] | np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray | float:
        batch_size = len(x_batch)

        tokenized_input = model.tokenizer.tokenize(x_batch)
        input_ids, attention_mask = unpack_token_ids_and_attention_mask(tokenized_input)
        x_batch_embeddings = safe_asarray(model.embedding_lookup(input_ids))

        scores = np.full((self.nr_samples, batch_size), fill_value=np.NINF)

        for step_id in range(self.nr_samples):
            if self.perturbation_type == PerturbationType.plain_text:
                scores[step_id] = self._evaluate_batch_step_plain_text_noise(
                    model, x_batch, y_batch, a_batch, x_batch_embeddings
                )
            if self.perturbation_type == PerturbationType.latent_space:
                a_batch_numerical = np.asarray([i[1] for i in a_batch])
                scores[step_id] = self._evaluate_batch_step_latent_space_noise(
                    model,
                    y_batch,
                    a_batch_numerical,
                    x_batch_embeddings,
                    attention_mask,
                )

        scores = self.aggregate_instances(scores)
        return self.aggregate_func(scores) if self.return_aggregate else scores

    def _evaluate_batch_step_plain_text_noise(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        x_batch_embeddings: np.ndarray,
    ) -> np.ndarray | float:
        batch_size = len(x_batch)
        # Perturb input.
        x_perturbed = self.perturb_func(x_batch, **self.perturb_func_kwargs)  # noqa
        warn_perturbation_caused_no_change(np.asarray(x_batch), np.asarray(x_perturbed))

        changed_prediction_indices = self.indexes_of_changed_predictions_plain_text(
            model, x_batch, x_perturbed
        )

        a_perturbed_perturbed = self.explain_func(
            model, x_perturbed, y_batch, **self.explain_func_kwargs  # noqa
        )
        a_perturbed_perturbed = self.normalise_a_batch(a_perturbed_perturbed)

        similarities = np.zeros(batch_size)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if instance_id in changed_prediction_indices:
                similarities[instance_id] = np.nan
                continue

            sensitivities = explanation_similarity(
                a_batch[instance_id],
                a_perturbed_perturbed[instance_id],
                self.similarity_func,
            )
            numerator = self.norm_numerator(sensitivities)
            denominator = self.norm_denominator(
                x_batch_embeddings[instance_id].flatten()
            )

            similarities[instance_id] = numerator / denominator

        return similarities

    def _evaluate_batch_step_latent_space_noise(
        self,
        model: TextClassifier,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        x_batch_embeddings: np.ndarray,
        attention_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        batch_size = len(x_batch_embeddings)
        # Perturb input.
        # fmt: off
        x_batch_embeddings_perturbed = self.perturb_func(x_batch_embeddings, **self.perturb_func_kwargs) # noqa
        # fmt: on
        warn_perturbation_caused_no_change(
            x_batch_embeddings, x_batch_embeddings_perturbed
        )

        changed_prediction_indices = self.indexes_of_changed_predictions_latent(
            model, x_batch_embeddings, x_batch_embeddings_perturbed, attention_mask
        )

        a_batch_perturbed = self.explain_func(
            model,
            x_batch_embeddings_perturbed,
            y_batch,
            attention_mask=attention_mask,  # noqa
            **self.explain_func_kwargs,  # noqa
        )
        a_batch_perturbed = self.normalise_a_batch(a_batch_perturbed)

        similarities = np.zeros(batch_size)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if instance_id in changed_prediction_indices:
                similarities[instance_id] = np.nan
                continue

            sensitivities = self.similarity_func(
                a_batch[instance_id],
                a_batch_perturbed[instance_id],  # noqa
            )

            numerator = self.norm_numerator(sensitivities)
            denominator = self.norm_denominator(
                x_batch_embeddings[instance_id].flatten()
            )

            similarities[instance_id] = numerator / denominator

        return similarities

    @abstractmethod
    def aggregate_instances(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError
