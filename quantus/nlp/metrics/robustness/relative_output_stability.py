from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from quantus.helpers.relative_stability import relative_output_stability_objective
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.functions.perturb_func import spelling_replacement
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import ExplainFn, Explanation, NormaliseFn, PerturbFn
from quantus.nlp.helpers.utils import get_scores, safe_as_array
from quantus.nlp.metrics.robustness.internal.relative_stability import RelativeStability


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
        nr_samples: int = 50,
    ):
        # TODO: docstring
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func_kwargs=normalise_func_kwargs,
            normalise_func=normalise_func,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            nr_samples=nr_samples,
        )

    def __call__(
        self,
        model: TextClassifier,
        x_batch: List[str],
        *,
        y_batch: Optional[np.ndarray] = None,
        a_batch: Optional[List[Explanation] | np.ndarray] = None,
        explain_func: ExplainFn = explain,
        explain_func_kwargs: Optional[Dict] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        # TODO: docstring
        return super().__call__(
            model,
            x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            batch_size=batch_size,
        )

    def compute_objective_plain_text(
        self,
        model: TextClassifier,
        x_batch: List[str],
        x_batch_perturbed: List[str],
        a_batch: List[Explanation],
        a_batch_perturbed: List[Explanation],
    ) -> np.ndarray:
        h_x = model.predict(x_batch)
        h_xs = model.predict(x_batch_perturbed)

        e_x = get_scores(a_batch)
        e_xs = get_scores(a_batch_perturbed)
        return relative_output_stability_objective(h_x, h_xs, e_x, e_xs)

    def compute_objective_latent_space(
        self,
        model: TextClassifier,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        predict_kwargs: Dict,
    ) -> np.ndarray:
        h_x = model(x_batch, **predict_kwargs)
        h_x = safe_as_array(h_x)
        h_xs = model(x_batch_perturbed, **predict_kwargs)
        h_xs = safe_as_array(h_xs)
        return relative_output_stability_objective(
            h_x, h_xs, a_batch, a_batch_perturbed
        )
