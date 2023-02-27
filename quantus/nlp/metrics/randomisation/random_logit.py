from __future__ import annotations

import random
from typing import List, Optional, Dict, Callable, no_type_check
import numpy as np

from quantus.functions.similarity_func import ssim
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation, SimilarityFn, NormaliseFn
from quantus.nlp.metrics.batched_metric import BatchedMetric
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.utils import explanations_batch_similarity


class RandomLogit(BatchedMetric):
    """
    Implementation of the Random Logit Metric by Sixt et al., 2020.

    The Random Logit Metric computes the distance between the original explanation and a reference explanation of
    a randomly chosen non-target class.

    References:
        1) Leon Sixt et al.: "When Explanations Lie: Why Many Modified BP
        Attributions Fail." ICML (2020): 9046-9057.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[NormaliseFn] = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        similarity_func: SimilarityFn = ssim,
        seed: int = 42,
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
            default_plot_func=default_plot_func,
        )
        self.num_classes = num_classes
        self.similarity_func = similarity_func
        self.seed = seed

    @no_type_check
    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str] | np.ndarray,
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        **kwargs,
    ) -> np.ndarray | float:
        np.random.seed(self.seed)
        y_off = [self.off_label_choice(i) for i in y_batch]
        y_off = np.asarray(y_off)

        # Explain against a random class.
        a_perturbed = self.explain_func(
            model,
            x_batch,
            y_off,
            **self.explain_func_kwargs,  # noqa
        )

        # Normalise and take absolute values of the attributions, if True.
        a_perturbed = self.normalise_a_batch(a_perturbed)
        scores = explanations_batch_similarity(
            a_batch, a_perturbed, self.similarity_func
        )
        return np.asarray(scores)

    def off_label_choice(self, y: int) -> int:
        all_labels = list(range(self.num_classes))
        del all_labels[y]
        random.seed(self.seed)
        return random.choice(all_labels)
