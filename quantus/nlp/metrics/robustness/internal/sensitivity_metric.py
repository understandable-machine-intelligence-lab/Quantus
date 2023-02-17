from __future__ import annotations

from quantus.nlp.metrics.robustness.internal.robustness_metric import RobustnessMetric
from abc import abstractmethod
import numpy as np
from typing import List, Callable, Optional
from quantus.functions.norm_func import fro_norm

from quantus.nlp.helpers.types import (
    Explanation,
    TextClassifier,
    PerturbationType,
)
from quantus.nlp.helpers.utils import (
    unpack_token_ids_and_attention_mask,
    safe_asarray,
)
from quantus.helpers.warn import warn_perturbation_caused_no_change


class SensitivityMetric(RobustnessMetric):
    """Shared implementation for Avg Sensitivity, Max Sesnitivity, Local Lipschitz Estimate"""

    def __init__(
        self,
        *args,
        norm_numerator: Callable[[np.ndarray], float] = fro_norm,
        norm_denominator: Callable[[np.ndarray], float] = fro_norm,
        nr_samples: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Save metric-specific attributes.
        self.nr_samples = nr_samples
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator

    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation] | np.ndarray,
    ) -> np.ndarray | float:
        batch_size = len(x_batch)

        tokenized_input = model.tokenizer.tokenize(x_batch)
        input_ids, attention_mask = unpack_token_ids_and_attention_mask(tokenized_input)
        x_batch_embeddings = safe_asarray(model.embedding_lookup(input_ids))

        similarities = np.zeros((self.nr_samples, batch_size))

        for step_id in range(self.nr_samples):
            if self.perturbation_type == PerturbationType.plain_text:
                similarities[step_id] = self._evaluate_batch_step_plain_text_noise(
                    model, x_batch, y_batch, a_batch, x_batch_embeddings
                )
            if self.perturbation_type == PerturbationType.latent_space:
                a_batch_numerical = np.asarray([i[1] for i in a_batch])
                similarities[step_id] = self._evaluate_batch_step_latent_space_noise(
                    model,
                    y_batch,
                    a_batch_numerical,
                    x_batch_embeddings,
                    attention_mask,
                )

        scores = self.aggregate_instances(similarities)
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
        x_perturbed = self.perturb_func(x_batch, **self.perturb_func_kwargs)
        warn_perturbation_caused_no_change(np.asarray(x_batch), np.asarray(x_perturbed))

        changed_prediction_indices = self.indexes_of_changed_predictions_plain_text(
            model, x_batch, x_perturbed
        )

        a_perturbed = self.explain_func(
            model, x_perturbed, y_batch, **self.explain_func_kwargs  # noqa
        )
        a_perturbed = self.normalise_a_batch(a_perturbed)

        similarities = np.zeros(batch_size)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if (
                self.return_nan_when_prediction_changes
                and instance_id in changed_prediction_indices
            ):
                similarities[instance_id] = np.nan
                continue

            sensitivities = self.compute_similarity_plain_text(
                a_batch[instance_id],
                a_perturbed[instance_id],
                x_batch[instance_id],
                x_perturbed[instance_id],
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
        x_batch_embeddings_perturbed = self.perturb_func(
            x_batch_embeddings, **self.perturb_func_kwargs
        )
        warn_perturbation_caused_no_change(
            x_batch_embeddings, x_batch_embeddings_perturbed
        )

        changed_prediction_indices = self.indexes_of_changed_predictions_latent(
            model, x_batch_embeddings, x_batch_embeddings_perturbed, attention_mask
        )

        a_perturbed = self.explain_func(
            model,
            x_batch_embeddings_perturbed,
            y_batch,
            attention_mask,  # noqa
            **self.explain_func_kwargs,  # noqa
        )

        similarities = np.zeros(batch_size)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if instance_id in changed_prediction_indices:
                similarities[instance_id] = np.nan
                continue

            sensitivities = self.compute_similarity_latent_space(
                a_batch[instance_id],
                a_perturbed[instance_id],
                x_batch_embeddings[instance_id],
                x_batch_embeddings_perturbed[instance_id],
            )

            numerator = self.norm_numerator(sensitivities)
            denominator = self.norm_denominator(
                x_batch_embeddings[instance_id].flatten()
            )

            similarities[instance_id] = numerator / denominator

        return similarities

    @abstractmethod
    def compute_similarity_plain_text(
        self,
        a: Explanation,
        a_perturbed: Explanation,
        x: str,
        x_perturbed: str,
        model: TextClassifier,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_similarity_latent_space(
        self,
        a: np.ndarray,
        a_perturbed: np.ndarray,
        x: np.ndarray,
        x_perturbed: np.ndarray,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def aggregate_instances(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError
