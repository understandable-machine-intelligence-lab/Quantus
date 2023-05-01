from __future__ import annotations

from typing import Callable
from quantus.metrics.base_batched import BatchedPerturbationMetric
from quantus.helpers import collection_utils


class RobustnessMetricChain(BatchedPerturbationMetric):
    def __init__(
        self,
        metrics_factory: dict[str, Callable[[...], BatchedPerturbationMetric]],
        abs: bool = False,
        normalise: bool = False,
        normalise_func=None,
        normalise_func_kwargs=None,
        perturb_func=None,
        perturb_func_kwargs=None,
        return_aggregate=None,
        aggregate_func=None,
        default_plot_func=None,
        disable_warnings=True,
        display_progressbar=True,
        nr_samples: int = None,
        return_nan_when_prediction_changes: bool = None,
    ):
        super().__init__(
            abs,
            normalise,
            normalise_func,
            normalise_func_kwargs,
            perturb_func,
            perturb_func_kwargs,
            return_aggregate,
            aggregate_func,
            default_plot_func,
            disable_warnings,
            display_progressbar,
            nr_samples,
            return_nan_when_prediction_changes,
        )
        self.metrics = collection_utils.map_dict(
            metrics_factory,
            lambda cls: cls(
                abs,
                normalise,
                normalise_func,
                normalise_func_kwargs,
                perturb_func,
                perturb_func_kwargs,
                return_aggregate,
                aggregate_func,
                default_plot_func,
                disable_warnings,
                display_progressbar,
                nr_samples,
                return_nan_when_prediction_changes,
            ),
        )

    def evaluate_sample(
        self, model, x_batch, x_perturbed, a_batch, a_perturbed, y_batch, predict_kwargs
    ):
        result = {}
        for k, v in self.metrics.items():
            result[k] = v.evaluate_sample(
                model,
                x_batch,
                x_perturbed,
                a_batch,
                a_perturbed,
                y_batch,
                predict_kwargs,
            )
        return result

    def reduce_samples(self, scores: dict):
        result = {}
        for k, v in self.metrics.items():
            result[k] = v.join_batches(scores[k])

    @staticmethod
    def join_batches(score_batches):
        return collection_utils.join_dict(score_batches)
