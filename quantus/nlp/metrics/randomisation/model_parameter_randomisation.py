from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm
from functools import partial
from typing import List, Optional, Dict, Callable
from quantus.functions.similarity_func import correlation_spearman
from quantus.nlp.helpers.types import ExplainFn, Explanation, NormaliseFn, SimilarityFn
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.metrics.batched_metric import BatchedMetric
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.utils import (
    explanations_similarity,
    normalise_attributions,
    abs_attributions,
)


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
        similarity_func: SimilarityFn = correlation_spearman,
        layer_order: str = "independent",
        seed: int = 42,
        return_sample_correlation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.layer_order = layer_order
        self.return_sample_correlation = return_sample_correlation
        self.similarity_func = similarity_func

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
    ) -> List[np.ndarray | float]:
        data = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
        )
        model = data["model"]
        x_batch = data["x_batch"]
        y_batch = data["y_batch"]
        a_batch = data["a_batch"]

        # Results are returned/saved as a dictionary not as a list as in the super-class.
        self.last_results = {}

        # Get number of iterations from number of layers.
        n_layers = model.nr_layers

        model_iterator = tqdm(
            model.get_random_layer_generator(order=self.layer_order, seed=self.seed),
            total=n_layers,
            disable=not self.display_progressbar,
        )

        for layer_name, random_layer_model in model_iterator:
            # Generate an explanation with perturbed model.
            a_batch_perturbed = self.explain_func(
                random_layer_model,
                x_batch,
                y_batch,
                **self.explain_func_kwargs,
            )

            similarity_scores = self.evaluate_batch(a_batch, a_batch_perturbed)
            # Save similarity scores in a result dictionary.
            self.last_results[layer_name] = similarity_scores

        if self.return_sample_correlation:
            raise NotImplementedError

        if self.return_aggregate:
            assert self.return_sample_correlation, (
                "You must set 'return_average_correlation_per_sample'"
                " to True in order to compute te aggregat"
            )
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results

    def evaluate_batch(
        self,
        a_batch: List[Explanation],
        a_perturbed: List[Explanation],
    ) -> float:
        if self.normalise:
            normalise_fn = partial(self.normalise_func, **self.normalise_func_kwargs)
            a_perturbed = normalise_attributions(a_perturbed, normalise_fn)

        if self.abs:
            a_perturbed = abs_attributions(a_perturbed)

        # Compute distance measure.
        similarity_fn = partial(self.similarity_func, **self.similarity_func_kwargs)
        return explanations_similarity(a_batch, a_perturbed, similarity_fn)
