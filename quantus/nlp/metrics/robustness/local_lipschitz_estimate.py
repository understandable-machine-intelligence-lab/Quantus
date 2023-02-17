from __future__ import annotations

import numpy as np

from quantus.functions.similarity_func import lipschitz_constant
from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.metrics.robustness.internal.sensitivity_metric import SensitivityMetric
from quantus.nlp.helpers.utils import get_embeddings


class LocalLipschitzEstimate(SensitivityMetric):
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

    def compute_similarity_plain_text(
        self,
        a: Explanation,
        a_perturbed: Explanation,
        x: str,
        x_perturbed: str,
        model: TextClassifier,
    ) -> float:
        return self.compute_similarity_latent_space(
            a[1],
            a_perturbed[1],
            get_embeddings([x], model)[0],
            get_embeddings([x_perturbed], model)[0],
        )

    def compute_similarity_latent_space(
        self,
        a: np.ndarray,
        a_perturbed: np.ndarray,
        x: np.ndarray,
        x_perturbed: np.ndarray,
    ) -> float:
        return lipschitz_constant(
            a=a,
            b=a_perturbed,
            c=x,
            d=x_perturbed,
            norm_numerator=self.norm_numerator,
            norm_denominator=self.norm_denominator,
        )

    def aggregate_instances(self, scores: np.ndarray) -> np.ndarray:
        max_func = np.max if self.return_nan_when_prediction_changes else np.nanmax
        return max_func(scores, axis=1)
