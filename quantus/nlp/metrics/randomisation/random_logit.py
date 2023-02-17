from __future__ import annotations

import random
from typing import List, Optional, Dict, Callable
from functools import partial

import numpy as np

from quantus.functions.similarity_func import ssim
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation, NormaliseFn, SimilarityFn
from quantus.nlp.metrics.batched_metric import BatchedMetric
from quantus.nlp.helpers.utils import (
    normalise_attributions,
    abs_attributions,
    explanation_similarity,
)


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
        similarity_func: SimilarityFn = ssim,
        num_classes: int = 2,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.similarity_func = similarity_func
        self.seed = seed

    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
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
            **self.explain_func_kwargs,
        )

        # Normalise and take absolute values of the attributions, if True.
        if self.normalise:
            normalise_fn = partial(self.normalise_func, **self.normalise_func_kwargs)
            a_perturbed = normalise_attributions(a_batch, normalise_fn)

        if self.abs:
            a_perturbed = abs_attributions(a_batch)

        similarity_fn = partial(self.similarity_func, **self.similarity_func_kwargs)
        scores = [
            explanation_similarity(a, b, similarity_fn)
            for a, b in zip(a_batch, a_perturbed)
        ]
        return np.asarray(scores)

    def off_label_choice(self, y: int) -> int:
        all_labels = list(range(self.nr_classes))
        del all_labels[y]
        random.seed(self.seed)
        return random.choice(all_labels)
