from __future__ import annotations

from typing import List

import numpy as np

from quantus.nlp.helpers.types import Explanation
from quantus.metrics.robustness.internal.ris_objective import (
    RelativeInputStabilityObjective,
)
from quantus.nlp.metrics.robustness.internal.relative_stability import RelativeStability
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import get_embeddings


class RelativeInputStability(RelativeStability):
    """
    Relative Input Stability leverages the stability of an explanation with respect to the change in the input data

    :math:`RIS(x, x', e_x, e_x') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{x - x'}{x}||_p, \epsilon_{min})}`

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/abs/2203.06877
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective = RelativeInputStabilityObjective(self.eps_min)

    def compute_objective_latent_space(
        self,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        *args,
    ):
        return self.objective(x_batch, x_batch_perturbed, a_batch, a_batch_perturbed)

    def compute_objective_plain_text(
        self,
        x_batch: List[str],
        x_batch_perturbed: List[str],
        a_batch: List[Explanation],
        a_batch_perturbed: List[Explanation],
        model: TextClassifier,
    ) -> np.ndarray:
        x = get_embeddings(x_batch, model)
        xs = get_embeddings(x_batch_perturbed, model)

        return self.objective(
            x,
            xs,
            np.asarray([i[1] for i in a_batch]),
            np.asarray([i[1] for i in a_batch_perturbed]),
        )
