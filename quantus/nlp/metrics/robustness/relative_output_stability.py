from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Callable

from quantus.metrics.robustness.internal.ros_objective import (
    RelativeOutputStabilityObjective,
)

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import (
    Explanation,
    NormaliseFn,
    PerturbationType,
    NumericalPerturbFn,
    PlainTextPerturbFn,
)

from quantus.nlp.metrics.robustness.internal.relative_stability import RelativeStability
from quantus.nlp.helpers.utils import safe_asarray, pad_ragged_vector
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.functions.perturb_func import spelling_replacement


class RelativeOutputStability(RelativeStability):
    """
    Relative Output Stability leverages the stability of an explanation with respect
    to the change in the output logits

    :math:`ROS(x, x', ex, ex') = max \\frac{||\\frac{e_x - e_x'}{e_x}||_p}{max (||h(x) - h(x')||_p, \epsilon_{min})}`,

    where `h(x)` and `h(x')` are the output logits for `x` and `x'` respectively


    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/pdf/2203.06877.pdf
    """

    def __init__(
        self,
        abs: bool = False,  # noqa
        normalise: bool = True,
        normalise_func: NormaliseFn = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        perturbation_type: PerturbationType = PerturbationType.plain_text,
        perturb_func: PlainTextPerturbFn | NumericalPerturbFn = spelling_replacement,
        perturb_func_kwargs: Optional[Dict] = None,
        eps_min: float = 1e-5,
        nr_samples: int = 50,
        return_nan_when_prediction_changes: bool = False,
        default_plot_func: Optional[Callable] = None,
    ):
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func_kwargs=normalise_func_kwargs,
            normalise_func=normalise_func,
            return_aggregate=return_aggregate,
            return_nan_when_prediction_changes=return_nan_when_prediction_changes,
            aggregate_func=aggregate_func,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
            perturbation_type=perturbation_type,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            eps_min=eps_min,
            nr_samples=nr_samples,
            default_plot_func=default_plot_func,
        )
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

        e_x = np.asarray([i[1] for i in a_batch])
        e_xs = np.asarray([i[1] for i in a_batch_perturbed])

        e_x, e_xs = pad_ragged_vector(e_x, e_xs)
        return self.objective(h_x, h_xs, e_x, e_xs)
