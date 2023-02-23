from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Dict

from quantus.functions.norm_func import fro_norm
from quantus.functions.similarity_func import difference
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.types import (
    SimilarityFn,
    NormaliseFn,
    NormFm,
    PerturbationType,
    PlainTextPerturbFn,
    NumericalPerturbFn,
)
from quantus.nlp.functions.perturb_func import spelling_replacement
from quantus.nlp.metrics.robustness.internal.sensitivity_metric import SensitivityMetric


class AvgSensitivity(SensitivityMetric):
    """
    Implementation of Avg-Sensitivity by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuring how explanations
    change under slight perturbation - the average sensitivity is captured.

    References:
        1) Chih-Kuan Yeh et al. "On the (in) fidelity and sensitivity for explanations."
        NeurIPS (2019): 10965-10976.
        2) Umang Bhatt et al.: "Evaluating and aggregating
        feature-based model explanations."  IJCAI (2020): 3016-3022.
    """

    def __init__(
        self,
        *,
        similarity_func: SimilarityFn = difference,
        abs: bool = False,  # noqa
        normalise: bool = True,
        normalise_func: NormaliseFn = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        perturbation_type: PerturbationType = PerturbationType.plain_text,
        perturb_func: PlainTextPerturbFn | NumericalPerturbFn = spelling_replacement,
        perturb_func_kwargs: Optional[Dict] = None,
        norm_numerator: NormFm = fro_norm,
        norm_denominator: NormFm = fro_norm,
        nr_samples: int = 50,
        return_nan_when_prediction_changes: bool = False,
        default_plot_func: Optional[Callable] = None,
    ):
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
            perturbation_type=perturbation_type,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            norm_numerator=norm_numerator,
            norm_denominator=norm_denominator,
            nr_samples=nr_samples,
            return_nan_when_prediction_changes=return_nan_when_prediction_changes,
            similarity_func=similarity_func,
            default_plot_func=default_plot_func,
        )

    def aggregate_instances(self, scores: np.ndarray) -> np.ndarray:
        agg_fn = np.mean if self.return_nan_when_prediction_changes else np.nanmean
        return agg_fn(scores, axis=0)
