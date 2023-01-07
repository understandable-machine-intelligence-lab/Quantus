from __future__ import annotations

import numpy as np

from typing import List, Callable, Optional, Dict, Tuple

from quantus.nlp.functions.perturbation_function import spelling_replacement
from quantus.helpers.asserts import attributes_check
from quantus.nlp.helpers.types import ExplainFn, PerturbFn, SimilarityFn, NormaliseFn
from quantus.nlp.metrics.text_classification_metric import (
    BatchedTextClassificationMetric,
)
from nlp.helpers.types import TextClassifier


class AvgSensitivity(BatchedTextClassificationMetric):
    @attributes_check
    def __init__(
        self,
        similarity_func: Optional[SimilarityFn] = None,
        norm_numerator: Optional[Callable] = None,
        norm_denominator: Optional[Callable] = None,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[NormaliseFn] = None,
        normalise_func_kwargs: Optional[Dict] = None,
        perturb_func: PerturbFn = None,
        perturb_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func: callable
            Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=difference.
        norm_numerator: callable
            Function for norm calculations on the numerator.
            If None, the default value is used, default=fro_norm
        norm_denominator: callable
            Function for norm calculations on the denominator.
            If None, the default value is used, default=fro_norm
        nr_samples: integer
            The number of samples iterated, default=200.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=gaussian_noise.
        perturb_std: float
            The amount of noise added, default=0.1.
        perturb_mean: float
            The mean of noise added, default=0.0.
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        return_nan_when_prediction_changes: boolean
            When set to true, the metric will be evaluated to NaN if the prediction changes after the perturbation is applied.
        kwargs: optional
            Keyword arguments.
        """
        if normalise_func is None:
            normalise_func = normalise_by_max

        if perturb_func is None:
            perturb_func = uniform_noise

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["lower_bound"] = lower_bound
        perturb_func_kwargs["upper_bound"] = upper_bound

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        self.nr_samples = nr_samples

        if similarity_func is None:
            similarity_func = difference
        self.similarity_func = similarity_func

        if norm_numerator is None:
            norm_numerator = norm_func.fro_norm
        self.norm_numerator = norm_numerator

        if norm_denominator is None:
            norm_denominator = norm_func.fro_norm
        self.norm_denominator = norm_denominator
        self.return_nan_when_prediction_changes = return_nan_when_prediction_changes

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'lower_bound' and 'upper_bound', the number of samples "
                    "iterated over 'nr_samples', the function to perturb the input "
                    "'perturb_func', the similarity metric 'similarity_func' as well as "
                    "norm calculations on the numerator and denominator of the sensitivity"
                    " equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Yeh, Chih-Kuan, et al. 'On the (in) fidelity and sensitivity for explanations"
                    ".' arXiv preprint arXiv:1901.09392 (2019)"
                ),
            )
            warn.warn_noise_zero(noise=lower_bound)
