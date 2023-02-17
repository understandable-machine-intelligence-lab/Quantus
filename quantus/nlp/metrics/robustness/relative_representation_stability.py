from __future__ import annotations

from typing import List, Optional

import numpy as np

from quantus.metrics.robustness.internal.rrs_objective import (
    RelativeRepresentationStabilityObjective,
)
from quantus.nlp.helpers.types import Explanation, TextClassifier
from quantus.nlp.metrics.robustness.internal.relative_stability import RelativeStability


class RelativeRepresentationStability(RelativeStability):
    """
    Relative Output Stability leverages the stability of an explanation with respect
    to the change in the output logits

    :math:`RRS(x, x', ex, ex') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{L_x - L_{x'}}{L_x}||_p, \epsilon_{min})},`

    where `L(·)` denotes the internal model representation, e.g., output embeddings of hidden layers.

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/pdf/2203.06877.pdf
    """

    def __init__(
        self,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.objective = RelativeRepresentationStabilityObjective(self.eps_min)
        self.layer_names = layer_names
        self.layer_indices = layer_indices

    def compute_objective_plain_text(
        self,
        x_batch: List[str],
        x_batch_perturbed: List[str],
        a_batch: List[Explanation],
        a_batch_perturbed: List[Explanation],
        model: TextClassifier,
    ) -> np.ndarray:
        l_x = model.get_hidden_representations(
            x_batch, self.layer_names, self.layer_indices
        )
        l_xs = model.get_hidden_representations(
            x_batch_perturbed, self.layer_names, self.layer_indices
        )

        return self.objective(
            l_x,
            l_xs,
            np.asarray([i[1] for i in a_batch]),
            np.asarray([i[1] for i in a_batch_perturbed]),
        )

    def compute_objective_latent_space(
        self,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        model: TextClassifier,
        attention_mask: Optional[np.ndarray],
    ):
        l_x = model.get_hidden_representations_embeddings(
            x_batch, attention_mask, self.layer_names, self.layer_indices
        )
        l_xs = model.get_hidden_representations_embeddings(
            x_batch_perturbed, attention_mask, self.layer_names, self.layer_indices
        )
        return self.objective(l_x, l_xs, a_batch, a_batch_perturbed)
