from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Callable

from quantus.helpers.relative_stability import relative_output_stability_objective

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation, NormaliseFn, PerturbFn

from quantus.nlp.metrics.robustness.internal.relative_stability import RelativeStability
from quantus.nlp.helpers.utils import pad_ragged_arrays
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
        perturb_func: PerturbFn = spelling_replacement,
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
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            eps_min=eps_min,
            nr_samples=nr_samples,
            default_plot_func=default_plot_func,
        )

    def compute_objective_latent_space(
        self,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        model: TextClassifier,
        **kwargs,
    ):
        h_x = model(x_batch, **kwargs)
        h_x = safe_as_array(h_x)
        h_xs = model(x_batch_perturbed, **kwargs)
        h_xs = safe_as_array(h_xs)
        return relative_output_stability_objective(
            h_x, h_xs, a_batch, a_batch_perturbed, eps_min=self._eps_min
        )

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

        e_x, e_xs = pad_ragged_arrays(e_x, e_xs)
        return relative_output_stability_objective(
            h_x, h_xs, e_x, e_xs, eps_min=self._eps_min
        )
