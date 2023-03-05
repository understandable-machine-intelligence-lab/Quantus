from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from quantus.functions.similarity_func import ssim
from quantus.helpers.utils import off_label_choice
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import ExplainFn, Explanation, NormaliseFn, SimilarityFn
from quantus.nlp.helpers.utils import get_scores
from quantus.nlp.metrics.text_classification_metric import TextClassificationMetric


class RandomLogit(TextClassificationMetric):
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
    ):
        # TODO: docstring
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
        self.num_classes = num_classes
        self.similarity_func = similarity_func
        self.seed = seed

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
    ) -> np.ndarray:
        return super().__call__(
            model,
            x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            batch_size=batch_size,
        )

    def evaluate_batch(  # type: ignore
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
        **kwargs,
    ) -> np.ndarray | float:
        np.random.seed(self.seed)
        y_off = off_label_choice(y_batch, self.num_classes)

        # Explain against a random class.
        a_perturbed = self.explain_batch(
            model, x_batch, y_off, explain_func, explain_func_kwargs
        )
        return self.similarity_func(get_scores(a_batch), get_scores(a_perturbed))
