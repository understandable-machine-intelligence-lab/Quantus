"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, Dict, List, TYPE_CHECKING, Mapping, Tuple, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from quantus.functions.explanation_func import explain
from quantus.helpers import asserts
from quantus.helpers import utils
from quantus.helpers import warn
from quantus.helpers.collection_utils import (
    value_or_default,
    batch_inputs,
    flatten,
)
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.nlp_utils import map_explanations
from quantus.helpers.tf_utils import is_tensorflow_model
from quantus.helpers.types import Explanation, ExplainFn, PersistFn
from quantus.metrics.base_batched import BatchedMetric

if TYPE_CHECKING:
    from quantus.helpers.types import ModelT, TokenizerT


def evaluate(
    metrics: Dict,
    xai_methods: Union[Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]],
    model: ModelInterface,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    s_batch: Union[np.ndarray, None] = None,
    agg_func: Callable = lambda x: x,
    progress: bool = False,
    explain_func_kwargs: Optional[dict] = None,
    call_kwargs: Union[Dict, Dict[str, Dict]] = None,
    **kwargs,
) -> Optional[dict]:
    """
    A method to evaluate some explanation methods given some metrics.

    Parameters
    ----------
    metrics: dict
        A dictionary with intialised metrics.
    xai_methods: dict, list
        Pass the different explanation methods as:
        1) Dict[str, np.ndarray] where values are pre-calculcated attributions, or
        2) Dict[str, Dict] where the keys are the name of the Quantus build-in explanation methods,
        and the values are the explain function keyword arguments as a dictionary, or
        3) Dict[str, Callable] where the keys are the name of explanation methods,
        and the values a callable explanation function.
    model: torch.nn.Module, tf.keras.Model
        A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
    x_batch: np.ndarray
        A np.ndarray which contains the input data that are explained.
    y_batch: np.ndarray
        A np.ndarray which contains the output labels that are explained.
    s_batch: np.ndarray, optional
        A np.ndarray which contains segmentation masks that matches the input.
    agg_func: callable
        Indicates how to aggregates scores e.g., pass np.mean.
    progress: boolean
        Indicates if progress should be printed to std, or not.
    explain_func_kwargs: dict, optional
        Keyword arguments to be passed to explain_func on call. Pass None if using Dict[str, Dict] type for xai_methods.
    call_kwargs: Dict[str, Dict]
        Keyword arguments for the call of the metrics, keys are names for arg set and values are argument dictionaries.
    kwargs: optional
        Deprecated keyword arguments for the call of the metrics.
    Returns
    -------
    results: dict
        A dictionary with the results.
    """

    warn.check_kwargs(kwargs)

    if xai_methods is None:
        print("Define the explanation methods that you want to evaluate.")
        return None

    if metrics is None:
        print(
            "Define the Quantus evaluation metrics that you want to evaluate the explanations against."
        )
        return None

    if call_kwargs is None:
        call_kwargs = {}
    elif not isinstance(call_kwargs, Dict):
        raise TypeError("xai_methods type is not Dict[str, Dict].")

    results: Dict[str, dict] = {}
    explain_funcs: Dict[str, Callable] = {}

    if not isinstance(xai_methods, dict):
        "xai_methods type is not in: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."

    for method, value in xai_methods.items():
        results[method] = {}

        if callable(value):
            explain_funcs[method] = value
            explain_func = value

            # Asserts.
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **{**explain_func_kwargs, **{"method": method}},
            )
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)

            # Asserts.
            asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        elif isinstance(value, Dict):
            if explain_func_kwargs is not None:
                warnings.warn(
                    "Passed explain_func_kwargs will be ignored when passing type Dict[str, Dict] as xai_methods."
                    "Pass explanation arguments as dictionary values."
                )

            explain_func_kwargs = value
            explain_funcs[method] = explain

            # Generate explanations.
            a_batch = explain(
                model=model, inputs=x_batch, targets=y_batch, **explain_func_kwargs
            )
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)

            # Asserts.
            asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        elif isinstance(value, np.ndarray):
            explain_funcs[method] = explain
            a_batch = value

        else:
            raise TypeError(
                "xai_methods type is not in: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."
            )

        if explain_func_kwargs is None:
            explain_func_kwargs = {}

        for metric, metric_func in metrics.items():
            results[method][metric] = {}

            for call_kwarg_str, call_kwarg in call_kwargs.items():
                if progress:
                    print(
                        f"Evaluating {method} explanations on {metric} metric on set of call parameters {call_kwarg_str}..."
                    )

                results[method][metric][call_kwarg_str] = agg_func(
                    metric_func(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        s_batch=s_batch,
                        explain_func=explain_funcs[method],
                        explain_func_kwargs={
                            **explain_func_kwargs,
                            **{"method": method},
                        },
                        **call_kwarg,
                        **kwargs,
                    )
                )

    return results


class evaluate_text_classification(object):
    @staticmethod
    def varying_explain_func_kwargs(
        metric: BatchedMetric,
        model: ModelT,
        x_batch: List[str],
        y_batch: np.ndarray | None,
        explain_func: ExplainFn,
        explain_func_kwargs: Mapping[str, Dict[str, ...]],
        batch_size: int = 64,
        tokenizer: TokenizerT | None = None,
        verbose: bool = True,
        persist_callback: PersistFn | None = None,
    ) -> Dict[str, np.ndarray | float | Dict[str, ...]]:
        """One metric, different hyper parameters."""

        if "NLP" not in metric.data_domain_applicability:
            raise ValueError(f"{metric} does not support NLP.")

        if is_tensorflow_model(model):
            model_predict_kwargs = dict(batch_size=batch_size, verbose=0)
        else:
            model_predict_kwargs = dict()

        model_wrapper = utils.get_wrapped_text_classifier(
            model=model, model_predict_kwargs=model_predict_kwargs, tokenizer=tokenizer
        )
        if y_batch is None:
            y_batch = model_wrapper.predict(x_batch).argmax(axis=-1)

        result = {}
        pbar = tqdm(
            explain_func_kwargs.items(), disable=not verbose, desc="Evaluation..."
        )
        for k, v in pbar:  # noqa
            scores = metric(
                model=model_wrapper,
                x_batch=x_batch,
                y_batch=y_batch,
                # No pre-computed a-batch since it is expected to differ for every next XAI method.
                a_batch=None,
                explain_func=explain_func,
                explain_func_kwargs=v,
            )  # noqa

            if persist_callback is not None:
                persist_callback(k, v, scores)

            result[k] = scores
        return result

    @staticmethod
    def on_multiple_metrics(
        *,
        metrics: Mapping[str, BatchedMetric],
        model: ModelT,
        x_batch: List[str],
        y_batch: np.ndarray | None,
        explain_func: ExplainFn | None,
        explain_func_kwargs: Dict[str, ...] | None = None,
        batch_size: int = 64,
        tokenizer: TokenizerT | None = None,
        verbose: bool = True,
        persist_callback: PersistFn | None = None,
    ) -> Dict[str, np.ndarray | Dict[str, np.ndarray]]:
        """Evaluate set of metrics for a single set of explanation_func hyper parameters."""
        for i in metrics.values():
            if "NLP" not in i.data_domain_applicability:
                raise ValueError(f"{i} does not support NLP.")

        (
            model,
            y_batch,
            a_batch,
            metric_wise_a_batch,
        ) = evaluate_text_classification.prepare_text_classification_metrics_inputs(
            metrics=metrics,
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            batch_size=batch_size,
            tokenizer=tokenizer,
        )

        pbar = tqdm(
            total=len(metrics.keys()), disable=not verbose, desc="Evaluation..."
        )

        result = {}

        with pbar as pbar:
            for metric_name, metric_instance in metrics.items():
                pbar.desc = f"Evaluating {metric_name}"
                if metric_name in metric_wise_a_batch:
                    a_batch_for_metric = metric_wise_a_batch[metric_name]
                else:
                    a_batch_for_metric = a_batch

                scores = metric_instance(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch_for_metric,
                    explain_func=explain_func,
                    explain_func_kwargs=explain_func_kwargs,
                    batch_size=batch_size,
                )  # noqa
                result[metric_name] = scores

                if persist_callback is not None:
                    persist_callback(metric_name, explain_func_kwargs, scores)
                pbar.update()

        return result

    @staticmethod
    def prepare_text_classification_metrics_inputs(
        metrics: Mapping[str, BatchedMetric],
        model: ModelT,
        x_batch: List[str],
        y_batch: np.ndarray | None,
        explain_func: ExplainFn | None,
        explain_func_kwargs: Mapping[str, ...] | None,
        batch_size: int,
        tokenizer,
    ):
        if is_tensorflow_model(model):
            model_predict_kwargs = dict(batch_size=batch_size, verbose=0)
        else:
            model_predict_kwargs = dict()

        model_wrapper = utils.get_wrapped_text_classifier(
            model=model, model_predict_kwargs=model_predict_kwargs, tokenizer=tokenizer
        )
        if y_batch is None:
            y_batch = model_wrapper.predict(x_batch).argmax(axis=-1)
        explain_func_kwargs = value_or_default(explain_func_kwargs, lambda: {})

        x_batches = batch_inputs(x_batch, batch_size)
        y_batches = batch_inputs(y_batch, batch_size)

        a_batches = []
        for x, y in zip(x_batches, y_batches):
            # We need to batch already here, otherwise this would mean computing gradient over whole dataset.
            a = explain_func(model_wrapper, x, y, **explain_func_kwargs)
            a_batches.append(a)
        a_batch = flatten(a_batches)

        # Some metric may need normalization or absolute values.
        metric_wise_a_batch = {}
        for name, metric in metrics.items():
            a_batch_for_metric = None

            if metric.normalise:
                a_batch_for_metric = map_explanations(
                    a_batch,
                    partial(metric.normalise_func, **metric.normalise_func_kwargs),
                )
            if metric.abs:
                a_batch_for_metric = map_explanations(
                    value_or_default(a_batch_for_metric, lambda: a_batch), np.abs
                )

            if a_batch_for_metric is not None:
                metric_wise_a_batch[name] = a_batch_for_metric

        return model_wrapper, y_batch, a_batch, metric_wise_a_batch
