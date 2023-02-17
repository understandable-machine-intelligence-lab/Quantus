from __future__ import annotations

import numpy as np
from typing import List, Optional

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.utils import get_embeddings
from quantus.metrics.robustness.internal.ros_objective import (
    RelativeOutputStabilityObjective,
)
from quantus.nlp.metrics.robustness.internal.relative_stability import RelativeStability
from quantus.nlp.helpers.utils import safe_asarray


class RelativeOutputStability(RelativeStability):
    """
    Relative Output Stability leverages the stability of an explanation with respect
    to the change in the output logits

    :math:`ROS(x, x', ex, ex') = max \\frac{||\\frac{e_x - e_x'}{e_x}||_p}{max (||h(x) - h(x')||_p, \epsilon_{min})}`,

    where `h(x)` and `h(x')` are the output logits for `x` and `x'` respectively


    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/pdf/2203.06877.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective = RelativeOutputStabilityObjective(self.eps_min)

    def compute_objective_latent_space(
        self,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        model: TextClassifier,
        attention_mask: Optional[np.ndarray],
    ):
        h_x = model(x_batch, attention_mask)
        h_x = safe_asarray(h_x)
        h_xs = model(x_batch_perturbed, attention_mask)
        h_xs = safe_asarray(h_xs)
        return self.objective(h_x, h_xs, a_batch, a_batch_perturbed)

    def compute_objective_plain_text(
        self,
        x_batch: List[str],
        x_batch_perturbed: List[str],
        a_batch: List[Explanation],
        a_batch_perturbed: List[Explanation],
        model: TextClassifier,
    ) -> np.ndarray:
        h_x = model.predict(x_batch)
        h_xs = model.predict(x_batch_perturbed)

        return self.objective(
            h_x,
            h_xs,
            np.asarray([i[1] for i in a_batch]),
            np.asarray([i[1] for i in a_batch_perturbed]),
        )
