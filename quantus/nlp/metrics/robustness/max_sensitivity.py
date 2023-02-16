from __future__ import annotations

import numpy as np
from typing import List, Callable, Optional, Dict
import functools

from quantus.functions.similarity_func import difference
from quantus.helpers.asserts import attributes_check
from quantus.functions.norm_func import fro_norm

from quantus.nlp.helpers.types import (
    PlainTextPerturbFn,
    Explanation,
    SimilarityFn,
    NumericalPerturbFn,
    NormaliseFn,
    TextClassifier,
    PerturbationType,
)
from quantus.nlp.metrics.batched_text_classification_metric import (
    BatchedTextClassificationMetric,
)
from quantus.nlp.metrics.robustness.batched_robustness_metric import (
    BatchedRobustnessMetric,
)
from quantus.nlp.helpers.utils import (
    value_or_default,
    explanation_similarity,
    unpack_token_ids_and_attention_mask,
    safe_asarray,
)
from quantus.helpers.warn import warn_perturbation_caused_no_change
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.functions.perturb_func import spelling_replacement, uniform_noise


class MaxSensitivity(BatchedTextClassificationMetric, BatchedRobustnessMetric):  # noqa

    """
    Implementation of Max-Sensitivity by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuring how explanations
    change under slight perturbation - the average sensitivity is captured.

    References:
        1) Chih-Kuan Yeh et al. "On the (in) fidelity and sensitivity for explanations."
        NeurIPS (2019): 10965-10976.
        2) Umang Bhatt et al.: "Evaluating and aggregating
        feature-based model explanations."  IJCAI (2020): 3016-3022.
    """

    similarity_func: SimilarityFn
    similarity_func_kwargs: Dict

    @attributes_check
    def __init__(
        self,
        *,
        similarity_func: SimilarityFn = difference,
        similarity_func_kwargs: Optional[Dict] = None,
        norm_numerator: Callable[[np.ndarray], float] = fro_norm,
        norm_denominator: Callable[[np.ndarray], float] = fro_norm,
        nr_samples: int = 50,
        abs: bool = False,  # noqa
        normalise: bool = False,
        normalise_func: NormaliseFn = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        perturbation_type: PerturbationType = PerturbationType.plain_text,
        perturb_func: PlainTextPerturbFn | NumericalPerturbFn = None,
        perturb_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable[[np.ndarray], float] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
        default_plot_func: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func: Callable[[np.ndarray, np.ndarray], np.ndarray | float]
            Similarity function applied to compare input and perturbed input, default=difference.
        similarity_func_kwargs: Dict
            Kwargs passed to similarity_func, default=None.
        norm_numerator: Callable[[np.ndarray], float]
            Function for norm calculations on the numerator, default=fro_norm
        norm_denominator: Callable[[np.ndarray], float]
            Function for norm calculations on the numerator, default=fro_norm.
        nr_samples: int
            The number of samples iterated over, default=50.
        abs: bool
            Indicates, whether the absolute value of explanation scores should be used for evaluation, default=False.
        normalise: bool
            Indicates, whether the explanation scores should be normalised, default=False.
        normalise_func: Callable[[np.ndarray], np.ndarray]
            Function used to normalise explanation scores. Default `quantus.nlp.normalize_sum_to_1`.
        normalise_func_kwargs:
            Kwargs passed to normalise_func, default=None.
        perturbation_type:
            Plain text means perturb_func will be applied to plain text inputs.
            Latent means the perturb_func will be applied to sum of word and positional embeddings, default="plain_text".
        perturb_func: Callable[[List[str], ...], List[str]] | Callable[[np.ndarray, ...], np.ndarray]
            A function used to apply noise to inputs, default="spelling_replacement".
            Must have Callable[[List[str], ...], List[str]] signature for noise_type="plain_text".
            Must have Callable[[np.ndarray, ...], np.ndarray] signature for noise_type="latent".
            Alternatively, user can provide the name of 1 of available perturbation functions.
            All available function can be listed with `quantus.nlp.available_perturbation_functions()`.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: Callable[[np.ndarray], float]
            Callable that aggregates the scores given an evaluation call.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        return_nan_when_prediction_changes: bool
            When set to true, the metric will be evaluated to NaN if the prediction changes after the perturbation is applied.
        default_plot_func: callable
            Callable that plots the metrics result.
        kwargs:
            Unused.
        """

        if perturb_func is None:
            if perturbation_type == PerturbationType.plain_text:
                perturb_func = spelling_replacement
            if perturbation_type == PerturbationType.latent_space:
                perturb_func = uniform_noise

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

        # Save metric-specific attributes.
        self.noise_type = perturbation_type
        self.nr_samples = nr_samples
        self.similarity_func = similarity_func
        self.similarity_func_kwargs = value_or_default(
            similarity_func_kwargs, lambda: {}
        )
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator
        self.return_nan_when_prediction_changes = return_nan_when_prediction_changes

    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation] | np.ndarray,
        **kwargs,
    ) -> np.ndarray | float:
        """
        Parameters
        ----------
        model: `quantus.nlp.TextClassifier`
            Model that is subject to explanation.
        x_batch: List[str]
            A batch of plain text inputs.
        y_batch: np.ndarray
            A batch of labels for x_batch.
        a_batch: List[Tuple[List[str], np.ndarray]]
            A batch of explanations for x_batch and y_batch.
        kwargs:
            Unused

        Returns
        -------
        result: np.ndarray | float
            Returns float if return_aggregate=True, otherwise np.ndarray.
        """

        batch_size = len(x_batch)

        tokenized_input = model.tokenizer.tokenize(x_batch)
        input_ids, attention_mask = unpack_token_ids_and_attention_mask(tokenized_input)
        x_batch_embeddings = safe_asarray(model.embedding_lookup(input_ids))

        similarities = np.zeros((self.nr_samples, batch_size))

        for step_id in range(self.nr_samples):
            if self.noise_type == PerturbationType.plain_text:
                similarities[step_id] = self._evaluate_batch_step_plain_text_noise(
                    model, x_batch, y_batch, a_batch, x_batch_embeddings
                )
            if self.noise_type == PerturbationType.latent_space:
                a_batch_numerical = np.asarray([i[1] for i in a_batch])
                similarities[step_id] = self._evaluate_batch_step_latent_space_noise(
                    model, y_batch, a_batch_numerical, x_batch_embeddings, attention_mask
                )

        agg_fn = np.max if self.return_nan_when_prediction_changes else np.nanmax
        scores = agg_fn(similarities, axis=0)
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
        # Generate explanation based on perturbed input x.
        a_perturbed = self.explain_func(model, x_perturbed, y_batch, **self.explain_func_kwargs)
        a_perturbed = self.normalise_a_batch(a_perturbed)

        similarities = np.zeros(batch_size)
        similarity_fn = functools.partial(
            self.similarity_func, **self.similarity_func_kwargs
        )

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if (
                self.return_nan_when_prediction_changes
                and instance_id in changed_prediction_indices
            ):
                similarities[instance_id] = np.nan
                continue

            sensitivities = explanation_similarity(
                a_batch[instance_id],
                a_perturbed[instance_id],
                similarity_fn,  # noqa
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

        # Generate explanation based on perturbed input x.
        a_perturbed = self.explain_func(
            model, x_batch_embeddings_perturbed, y_batch, attention_mask, **self.explain_func_kwargs
        )
        a_perturbed = self.normalise_a_batch(a_perturbed)

        similarities = np.zeros(batch_size)
        similarity_fn = functools.partial(
            self.similarity_func, **self.similarity_func_kwargs
        )

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if (
                self.return_nan_when_prediction_changes
                and instance_id in changed_prediction_indices
            ):
                similarities[instance_id] = np.nan
                continue

            sensitivities = similarity_fn(
                a_batch[instance_id], a_perturbed[instance_id]
            )

            numerator = self.norm_numerator(sensitivities)
            denominator = self.norm_denominator(
                x_batch_embeddings[instance_id].flatten()
            )

            similarities[instance_id] = numerator / denominator

        return similarities
