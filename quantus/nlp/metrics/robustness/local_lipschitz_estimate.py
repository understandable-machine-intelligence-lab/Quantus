from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Callable, List, no_type_check

from quantus.helpers.warn import warn_perturbation_caused_no_change
from quantus.functions.similarity_func import lipschitz_constant, distance_euclidean
from quantus.nlp.functions.perturb_func import spelling_replacement
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.types import (
    Explanation,
    NormaliseFn,
    PerturbationType,
    PerturbFn,
)
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.metrics.robustness.internal.robustness_metric import RobustnessMetric
from quantus.nlp.helpers.utils import (
    get_embeddings,
    pad_ragged_arrays,
    determine_perturbation_type,
)


class LocalLipschitzEstimate(RobustnessMetric):
    """
    Implementation of the Local Lipschitz Estimate (or Stability) test by Alvarez-Melis et al., 2018a, 2018b.

    This tests asks how consistent are the explanations for similar/neighboring examples.
    The test denotes a (weaker) empirical notion of stability based on discrete,
    finite-sample neighborhoods i.e., argmax_(||f(x) - f(x')||_2 / ||x - x'||_2)
    where f(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) David Alvarez-Melis and Tommi S. Jaakkola. "On the robustness of interpretability methods."
        arXiv preprint arXiv:1806.08049 (2018).

        2) David Alvarez-Melis and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." NeurIPS (2018): 7786-7795.
    """

    def __init__(
        self,
        *,
        abs: bool = False,  # noqa
        normalise: bool = True,
        normalise_func: NormaliseFn = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        perturb_func: PerturbFn = spelling_replacement,
        perturb_func_kwargs: Optional[Dict] = None,
        nr_samples: int = 50,
        return_nan_when_prediction_changes: bool = False,
        default_plot_func: Optional[Callable] = None,
    ):
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            return_nan_when_prediction_changes=return_nan_when_prediction_changes,
            default_plot_func=default_plot_func,
        )
        self.nr_samples = nr_samples

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

        max_func = np.max if self.return_nan_when_prediction_changes else np.nanmax
        scores = max_func(scores, axis=0)
        return scores

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

        a_batch_perturbed = self.explain_func(
            model, x_perturbed, y_batch, **self.explain_func_kwargs  # noqa
        )
        a_batch_perturbed = self.normalise_a_batch(a_batch_perturbed)

        a_batch_scores = [i[1] for i in a_batch]
        a_batch_perturbed_scores = [i[1] for i in a_batch_perturbed]

        x_batch_embeddings_perturbed, _ = get_embeddings(x_perturbed, model)

        similarities = np.full(batch_size, fill_value=np.NINF)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if instance_id in changed_prediction_indices:
                similarities[instance_id] = np.nan
                continue

            a, a_batch_perturbed = pad_ragged_arrays(
                a_batch_scores[instance_id], a_batch_perturbed_scores[instance_id]
            )
            x, x_perturbed = pad_ragged_arrays(
                x_batch_embeddings[instance_id],
                x_batch_embeddings_perturbed[instance_id],
            )
            sensitivities = lipschitz_constant(
                a=a,
                b=a_batch_perturbed,
                c=x.reshape(-1),
                d=x_perturbed.reshape(-1),
                norm_numerator=distance_euclidean,
                norm_denominator=distance_euclidean,
            )

            similarities[instance_id] = sensitivities

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
        x_batch_embeddings_perturbed = self.perturb_func(x_batch_embeddings, **self.perturb_func_kwargs)  # noqa
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
        similarities = np.full(batch_size, fill_value=np.NINF)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if instance_id in changed_prediction_indices:
                similarities[instance_id] = np.nan
                continue

            a, a_perturbed = pad_ragged_arrays(
                a_batch_numerical[instance_id], a_batch_perturbed[instance_id]  # noqa
            )
            x, x_perturbed = pad_ragged_arrays(
                x_batch_embeddings[instance_id],
                x_batch_embeddings_perturbed[instance_id],
            )
            sensitivities = lipschitz_constant(
                a=a,
                b=a_perturbed,
                c=x.reshape(-1),
                d=x_perturbed.reshape(-1),
                norm_numerator=distance_euclidean,
                norm_denominator=distance_euclidean,
            )

            similarities[instance_id] = sensitivities

        return similarities
