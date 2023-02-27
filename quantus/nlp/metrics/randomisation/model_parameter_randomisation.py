from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Callable, no_type_check
from quantus.functions.similarity_func import correlation_spearman
from quantus.metrics.randomisation.model_parameter_randomisation import (
    ModelParameterRandomisation as BaseMPR,
)
from quantus.helpers.plotting import plot_model_parameter_randomisation_experiment
from quantus.nlp.helpers.types import ExplainFn, Explanation, NormaliseFn, SimilarityFn
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.metrics.batched_metric import BatchedMetric
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.utils import explanations_batch_similarity


class ModelParameterRandomisation(BatchedMetric):
    """
    Implementation of the Model Parameter Randomization Method by Adebayo et. al., 2018.

    The Model Parameter Randomization measures the distance between the original attribution and a newly computed
    attribution throughout the process of cascadingly/independently randomizing the model parameters of one layer
    at a time.

    Assumptions:
        - In the original paper multiple distance measures are taken: Spearman rank correlation (with and without abs),
        HOG and SSIM. We have set Spearman as the default value.

    References:
        1) Julius Adebayo et al.: "Sanity Checks for Saliency Maps."
        NeurIPS (2018): 9525-9536.
    """

    def __init__(
        self,
        *,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[NormaliseFn] = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        similarity_func: SimilarityFn = correlation_spearman,
        layer_order: str = "independent",
        seed: int = 42,
        return_sample_correlation: bool = False,
        default_plot_func: Optional[
            Callable
        ] = plot_model_parameter_randomisation_experiment,
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
            default_plot_func=default_plot_func,
        )
        self.seed = seed
        self.layer_order = layer_order
        self.return_sample_correlation = return_sample_correlation
        self.similarity_func = similarity_func

    @no_type_check
    def __call__(
        self,
        model: TextClassifier,
        x_batch: List[str],
        *,
        y_batch: Optional[np.ndarray] = None,
        a_batch: Optional[List[Explanation]] = None,
        explain_func: ExplainFn = explain,
        explain_func_kwargs: Optional[Dict] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> Dict[str, np.ndarray] | np.ndarray:
        data = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            batch_size=batch_size,
        )
        model = data["model"]
        x_batch = data["x_batch"]
        y_batch = data["y_batch"]
        a_batch = data["a_batch"]

        # Results are returned/saved as a dictionary not as a list as in the super-class.
        self.last_results = {}

        # Get number of iterations from number of layers.
        n_layers = len(
            list(
                model.get_random_layer_generator(
                    order=self.layer_order, seed=self.seed, **kwargs
                )
            )
        )

        model_iterator = tqdm(
            model.get_random_layer_generator(
                order=self.layer_order, seed=self.seed, **kwargs
            ),
            total=n_layers,
            disable=not self.display_progressbar,
        )

        for layer_name, random_layer_model in model_iterator:
            # Generate an explanation with perturbed model.
            a_batch_perturbed = self.explain_func(
                random_layer_model,
                x_batch,
                y_batch,
                **self.explain_func_kwargs,  # noqa
            )

            a_batch_perturbed = self.normalise_a_batch(a_batch_perturbed)

            similarity_scores = self.evaluate_batch(a_batch, a_batch_perturbed)
            # Save similarity scores in a result dictionary.
            self.last_results[layer_name] = similarity_scores

        if self.return_sample_correlation:
            return np.asarray(BaseMPR.compute_correlation_per_sample(self.last_results))

        if self.return_aggregate:
            assert self.return_sample_correlation, (
                "You must set 'return_average_correlation_per_sample'"
                " to True in order to compute te aggregate"
            )
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results  # type: ignore

    @no_type_check
    def evaluate_batch(
        self,
        a_batch: List[Explanation],
        a_perturbed: List[Explanation],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Compute distance measure.
        return explanations_batch_similarity(a_batch, a_perturbed, self.similarity_func)
