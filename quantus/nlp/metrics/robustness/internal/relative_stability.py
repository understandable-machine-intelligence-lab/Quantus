from __future__ import annotations

from typing import Optional, List, no_type_check
import numpy as np


from quantus.nlp.helpers.types import PerturbationType, TextClassifier, Explanation
from quantus.nlp.metrics.robustness.internal.robustness_metric import RobustnessMetric
from quantus.nlp.helpers.utils import unpack_token_ids_and_attention_mask, safe_asarray
from abc import abstractmethod


class RelativeStability(RobustnessMetric):
    """
    Relative Input Stability leverages the stability of an explanation with respect to the change in the input data

    :math:`RIS(x, x', e_x, e_x') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{x - x'}{x}||_p, \epsilon_{min})}`

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/abs/2203.06877
    """

    def __init__(self, *, eps_min: float, nr_samples: int, **kwargs):
        super().__init__(**kwargs)
        self.eps_min = eps_min
        self.nr_samples = nr_samples

    @no_type_check
    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        **kwargs,
    ) -> np.ndarray | float:
        batch_size = len(x_batch)
        scores = np.full((self.nr_samples, batch_size), fill_value=np.NINF)

        for step_id in range(self.nr_samples):
            if self.perturbation_type == PerturbationType.plain_text:
                scores[step_id] = self._evaluate_batch_step_plain_text_noise(
                    model, x_batch, y_batch, a_batch
                )
            if self.perturbation_type == PerturbationType.latent_space:
                a_batch_numerical = np.asarray([i[1] for i in a_batch])
                scores[step_id] = self._evaluate_batch_step_latent_space_noise(
                    model,
                    x_batch,
                    y_batch,
                    a_batch_numerical,
                )

        # Compute RIS.
        result = np.max(scores, axis=0)
        return result

    def _evaluate_batch_step_plain_text_noise(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
    ) -> np.ndarray | float:
        # Perturb input.
        x_perturbed = self.perturb_func(x_batch, **self.perturb_func_kwargs)  # noqa
        # Generate explanations for perturbed input.
        # fmt: off
        a_batch_perturbed = self.explain_func(model, x_perturbed, y_batch, **self.explain_func_kwargs)  # noqa
        # fmt: on
        a_batch_perturbed = self.normalise_a_batch(a_batch_perturbed)
        # Compute maximization's objective.
        rs = self.compute_objective_plain_text(
            x_batch, x_perturbed, a_batch, a_batch_perturbed, model
        )
        # We're done with this sample if `return_nan_when_prediction_changes`==False.
        if not self.return_nan_when_prediction_changes:
            return rs

        # If perturbed input caused change in prediction, then it's RIS=nan.
        changed_prediction_indices = self.indexes_of_changed_predictions_plain_text(
            model, x_batch, x_perturbed
        )

        if len(changed_prediction_indices) != 0:
            rs[changed_prediction_indices] = np.nan

        return rs

    def _evaluate_batch_step_latent_space_noise(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: np.ndarray,
    ) -> np.ndarray:
        tokenized_input = model.tokenizer.tokenize(x_batch)
        input_ids, attention_mask = unpack_token_ids_and_attention_mask(tokenized_input)
        x_batch_embeddings = safe_asarray(model.embedding_lookup(input_ids))

        # Perturb input.
        # fmt: off
        x_perturbed = self.perturb_func(x_batch_embeddings, **self.perturb_func_kwargs)  # noqa
        # fmt: on
        # Generate explanations for perturbed input.
        a_batch_perturbed = self.explain_func(
            model,
            x_perturbed,
            y_batch,
            attention_mask=attention_mask,  # noqa
            **self.explain_func_kwargs,  # noqa
        )
        a_batch_perturbed = self.normalise_a_batch(a_batch_perturbed)
        # Compute maximization's objective.
        rs = self.compute_objective_latent_space(
            x_batch_embeddings,
            x_perturbed,
            a_batch,
            a_batch_perturbed,  # noqa
            model,
            attention_mask,
        )
        # We're done with this sample if `return_nan_when_prediction_changes`==False.
        if not self.return_nan_when_prediction_changes:
            return rs

        # If perturbed input caused change in prediction, then it's RIS=nan.
        changed_prediction_indices = self.indexes_of_changed_predictions_latent(
            model, x_batch_embeddings, x_perturbed, attention_mask
        )

        if len(changed_prediction_indices) != 0:
            rs[changed_prediction_indices] = np.nan

        return rs

    @abstractmethod
    def compute_objective_latent_space(
        self,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        model: TextClassifier,
        attention_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_objective_plain_text(
        self,
        x_batch: List[str],
        x_batch_perturbed: List[str],
        a_batch: List[Explanation],
        a_batch_perturbed: List[Explanation],
        model: TextClassifier,
    ) -> np.ndarray:
        raise NotImplementedError
