from __future__ import annotations

from collections import defaultdict
from functools import partial
from operator import itemgetter
from typing import Callable, Dict, List, Optional, no_type_check

import numpy as np
from tqdm.auto import tqdm

from quantus.functions.similarity_func import correlation_spearman
from quantus.helpers.utils import compute_correlation_per_sample
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import ExplainFn, Explanation, NormaliseFn, SimilarityFn
from quantus.nlp.helpers.utils import (
    batch_list,
    get_scores,
    map_dict,
    map_optional,
    value_or_default,
)
from quantus.nlp.metrics.text_classification_metric import TextClassificationMetric


class ModelParameterRandomisation(TextClassificationMetric):
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
    ) -> Dict[str, np.ndarray | float] | np.ndarray:
        explain_func_kwargs = value_or_default(explain_func_kwargs, lambda: {})
        x_batch = batch_list(x_batch, batch_size)
        y_batch = map_optional(y_batch, partial(batch_list, batch_size=batch_size))
        a_batch = map_optional(a_batch, partial(batch_list, batch_size=batch_size))

        results = defaultdict(lambda: [])

        model_iterator = tqdm(
            model.get_random_layer_generator(self.layer_order, self.seed),
            total=model.random_layer_generator_length,
            disable=not self.display_progressbar,
        )

        for layer_name, random_layer_model in model_iterator:
            # Generate an explanation with perturbed model.
            for i, x in enumerate(x_batch):
                y = map_optional(y_batch, itemgetter(i))
                a = map_optional(a_batch, itemgetter(i))
                x, y, a, _ = self.batch_preprocess(
                    model, x, y, a, explain_func, explain_func_kwargs
                )
                similarity_score = self.evaluate_batch(
                    random_layer_model, x, y, a, explain_func, explain_func_kwargs
                )
                results[layer_name].extend(similarity_score)

        results = map_dict(results, np.asarray)

        if self.return_sample_correlation:
            return np.asarray(compute_correlation_per_sample(results))

        return results

    @no_type_check
    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
    ) -> np.ndarray:
        # Compute distance measure.
        a_batch_perturbed = self.explain_batch(
            model, x_batch, y_batch, explain_func, explain_func_kwargs
        )
        return self.similarity_func(get_scores(a_batch), get_scores(a_batch_perturbed))
