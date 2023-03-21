# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from collections import defaultdict
from functools import partial
from operator import itemgetter
from typing import Dict, List, Optional, no_type_check

import numpy as np
from tqdm.auto import tqdm

from quantus.helpers.utils import map_dict
from quantus.functions.similarity_func import correlation_spearman
from quantus.helpers.utils import compute_correlation_per_sample
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import (
    ExplainFn,
    Explanation,
    NormaliseFn,
    SimilarityFn,
)
from quantus.nlp.helpers.utils import (
    batch_list,
    get_scores,
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
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        similarity_func: SimilarityFn = correlation_spearman,
        layer_order: str = "independent",
        return_sample_correlation: bool = False,
    ):
        """

        Parameters
        ----------
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        similarity_func:
            Similarity function applied to compare explanations generated using original and randomized model.
        layer_order:
            Indicated whether the model is randomized cascadingly or independently.
            Set order=top_down for cascading randomization, set order=independent for independent randomization,
            default="independent".
        return_sample_correlation:
            Indicates whether return one float per sample, representing the average
            correlation coefficient across the layers for that sample.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
        )
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
        """

        Parameters
        ----------
        model:
            Torch or tensorflow model that is subject to explanation. Most probably, you will want to use
            `quantus.nlp.TorchHuggingFaceTextClassifier` or `quantus.nlp.TensorFlowHuggingFaceTextClassifier`,
            for out-of-the box support for models from Huggingface hub.
        x_batch:
            list, which contains the input data that are explained.
        y_batch:
            A np.ndarray which contains the output labels that are explained.
        a_batch:
            Pre-computed attributions i.e., explanations. Token and scores as well as scores only are supported.
        explain_func:
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        batch_size:
            Indicates size of batches, in which input dataset will be splitted.

        Returns
        -------

        score:
            np.ndarray of scores.

        """
        explain_func_kwargs = value_or_default(explain_func_kwargs, lambda: {})
        x_batch = batch_list(x_batch, batch_size)
        y_batch = map_optional(y_batch, partial(batch_list, batch_size=batch_size))
        a_batch = map_optional(a_batch, partial(batch_list, batch_size=batch_size))

        results = defaultdict(lambda: [])

        model_iterator = tqdm(
            model.get_random_layer_generator(self.layer_order),
            total=model.random_layer_generator_length,
            disable=not self.display_progressbar,
        )

        for layer_name, random_layer_model in model_iterator:
            # Generate an explanation with perturbed model.
            for i, x in enumerate(x_batch):
                y = map_optional(y_batch, itemgetter(i))
                a = map_optional(a_batch, itemgetter(i))
                x, y, a, _ = self._batch_preprocess(
                    model, x, y, a, explain_func, explain_func_kwargs
                )
                similarity_score = self._evaluate_batch(
                    random_layer_model, x, y, a, explain_func, explain_func_kwargs
                )
                results[layer_name].extend(similarity_score)

        results = map_dict(results, np.asarray)

        if self.return_sample_correlation:
            return np.asarray(compute_correlation_per_sample(results))

        return results

    @no_type_check
    def _evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
    ) -> np.ndarray:
        # Compute distance measure.
        a_batch_perturbed = self._explain_batch(
            model, x_batch, y_batch, explain_func, explain_func_kwargs
        )
        return self.similarity_func(get_scores(a_batch), get_scores(a_batch_perturbed))
