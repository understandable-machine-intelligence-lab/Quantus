from __future__ import annotations

from typing import List, Optional, Callable, Dict

import numpy as np
from quantus.metrics.robustness.internal.ris_objective import (
    RelativeInputStabilityObjective,
)

from quantus.nlp.helpers.types import (
    Explanation,
    NormaliseFn,
    PlainTextPerturbFn,
    NumericalPerturbFn,
    PerturbationType,
)

from quantus.nlp.metrics.robustness.internal.relative_stability import RelativeStability
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import get_embeddings, pad_ragged_vector
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.functions.perturb_func import spelling_replacement


class RelativeInputStability(RelativeStability):
    """
    Relative Input Stability leverages the stability of an explanation with respect to the change in the input data

    :math:`RIS(x, x', e_x, e_x') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{x - x'}{x}||_p, \epsilon_{min})}`

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/abs/2203.06877
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
        self.objective = RelativeInputStabilityObjective(self.eps_min)

    def compute_objective_latent_space(
        self,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        *args,
        **kwargs,
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
        x, _ = get_embeddings(x_batch, model)
        xs, _ = get_embeddings(x_batch_perturbed, model)
        x, xs = pad_ragged_vector(x, xs)

        e_x = np.asarray([i[1] for i in a_batch])
        e_xs = np.asarray([i[1] for i in a_batch_perturbed])

        e_x, e_xs = pad_ragged_vector(e_x, e_xs)

        return self.objective(x, xs, e_x, e_xs)