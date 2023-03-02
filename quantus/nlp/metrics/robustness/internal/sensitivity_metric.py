from __future__ import annotations

from abc import abstractmethod
import numpy as np
from typing import List, Callable, Optional, no_type_check, Dict


from quantus.functions.similarity_func import difference
from quantus.nlp.helpers.types import (
    Explanation,
    TextClassifier,
    PerturbationType,
    SimilarityFn,
    NormFn,
)
from quantus.nlp.helpers.utils import (
    explanation_similarity,
    get_embeddings,
    determine_perturbation_type,
    flatten,
)
from quantus.helpers.warn import warn_perturbation_caused_no_change

from quantus.nlp.metrics.robustness.internal.robustness_metric import RobustnessMetric


class SensitivityMetric(RobustnessMetric):
    """Shared implementation for Avg Sensitivity, Max Sesnitivity, Local Lipschitz Estimate"""

    def __init__(
        self,
        *args,
        norm_numerator: NormFn,
        norm_denominator: NormFn,
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

    @no_type_check
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
        x_batch_embeddings, predict_kwargs = get_embeddings(x_batch, model)
        scores = np.full((self.nr_samples, batch_size), fill_value=np.NINF)
        perturbation_type = determine_perturbation_type(self.perturb_func)

        for step_id in range(self.nr_samples):
            if perturbation_type == PerturbationType.plain_text:
                scores[step_id] = self._evaluate_batch_step_plain_text_noise(
                    model, x_batch, y_batch, a_batch, x_batch_embeddings
                )
            if perturbation_type == PerturbationType.latent_space:
                scores[step_id] = self._evaluate_batch_step_latent_space_noise(
                    model,
                    y_batch,
                    a_batch,
                    x_batch_embeddings,
                    **predict_kwargs,
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

            sensitivities = explanation_similarity(  # type: ignore
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
        a_batch: List[Explanation],
        x_batch_embeddings: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        a_batch_numerical = np.asarray([i[1] for i in a_batch])
        batch_size = len(x_batch_embeddings)
        # Perturb input.
        # fmt: off
        x_batch_embeddings_perturbed = self.perturb_func(x_batch_embeddings, **self.perturb_func_kwargs) # noqa
        # fmt: on
        warn_perturbation_caused_no_change(
            x_batch_embeddings, x_batch_embeddings_perturbed
        )

        changed_prediction_indices = self.indexes_of_changed_predictions_latent(
            model, x_batch_embeddings, x_batch_embeddings_perturbed, **kwargs
        )

        a_batch_perturbed = self.explain_func(
            model,
            x_batch_embeddings_perturbed,
            y_batch,
            **self.explain_func_kwargs,  # noqa
            **kwargs,  # noqa
        )
        a_batch_perturbed = self.normalise_a_batch(a_batch_perturbed)

        similarities = np.zeros(batch_size)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if instance_id in changed_prediction_indices:
                similarities[instance_id] = np.nan
                continue

            sensitivities = self.similarity_func(
                a_batch_numerical[instance_id],
                a_batch_perturbed[instance_id],  # noqa
            )

            numerator = self.norm_numerator(sensitivities)
            denominator = self.norm_denominator(
                flatten(x_batch_embeddings[instance_id])
            )

            similarities[instance_id] = numerator / denominator

        return similarities

    @abstractmethod
    def aggregate_instances(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: not covered
