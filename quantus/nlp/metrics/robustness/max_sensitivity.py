from __future__ import annotations

import numpy as np
from quantus.functions.similarity_func import difference

from quantus.nlp.helpers.types import Explanation, SimilarityFn
from quantus.nlp.helpers.utils import explanation_similarity
from quantus.nlp.metrics.robustness.internal.sensitivity_metric import SensitivityMetric


class MaxSensitivityMetric(SensitivityMetric):

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

    def __init__(self, *, similarity_func: SimilarityFn = difference, **kwargs):
        super().__init__(**kwargs)

        # Save metric-specific attributes.
        self.similarity_func = similarity_func

    def compute_similarity_plain_text(
        self, a: Explanation, a_perturbed: Explanation, *args
    ) -> float:
        return explanation_similarity(a, a_perturbed, self.similarity_func)

    def compute_similarity_latent_space(
        self, a: np.ndarray, a_perturbed: np.ndarray, *args
    ) -> float:
        return self.similarity_func(a, a_perturbed)

    def aggregate_instances(self, scores: np.ndarray) -> np.ndarray:
        agg_fn = np.max if self.return_nan_when_prediction_changes else np.nanmax
        return agg_fn(scores, axis=0)
