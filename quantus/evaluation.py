"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import warnings
from typing import Union, Callable, Dict, Optional, List, Any
from collections import defaultdict
from tqdm.auto import tqdm

import numpy as np

from quantus.helpers import asserts
from quantus.helpers import utils
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.explanation_func import explain
from quantus.metrics.base_batched import BatchedMetric
from quantus.helpers.collection_utils import add_default_items


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


_MetricValue = Union[np.ndarray, Dict[str, np.ndarray]]
_CallKwargs = Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]
_PersistFn = Callable[[str, _CallKwargs, _MetricValue], None]


def evaluate_nlp(
    metrics: Dict[str, BatchedMetric],
    model,
    x_batch: List[str],
    y_batch: np.ndarray,
    verbose: bool = True,
    call_kwargs: Optional[_CallKwargs] = None,
    persist_callback: Optional[_PersistFn] = None,
) -> Dict[str, Union[_MetricValue, List[_MetricValue]]]:
    """

    Parameters
    ----------
     metrics:
        Dict where keys are any unique names, and values are pre-initialised metric instances.
    model:
        Model which is evaluated, must be a subclass of `qn.TextClassifer`
    x_batch:
        Batch of plain-text inputs for model.
    verbose:
        Indicates whether tqdm progress bar should be displayed.
    call_kwargs:
        kwargs, which are passed to metrics. Supported options are:
        - keys are metrics' names, and values are
        - Kwargs passed to each metrics' __call__ method. In this case each metric is evaluated once.
        - List of dicts. In this case each metric is evaluated with each entry of list of call_kwargs.
        - All key, values, where are not in metric names, kwargs are treated as global and passed to each metric.
        Metrics-specific kwargs will override global ones.
        Internally, `defaultdict` is used, so there is no need to provide kwargs for metrics,
            which should be evaluated with default ones.
    persist_callback:
        If passed, this function will be called after every metric is evaluated.
        It will be called with metric name, kwargs passed to __call__ of metric instance, and scores.
        This can be used to save intermideate results, e.g, in CSV file or database. Since complete evaluation
        can take long, and loosing intermideate results can be annoying ;(.



    Returns
    -------

    scores:
        Dict, keys are metric names, values are scores, or list of scores for corresponding metric.

    """

    # TODO:
    #  - batch inputs
    #  - generate a_batch, y_batch

    for i in metrics.values():
        if "NLP" not in i.data_domain_applicability:
            raise ValueError(f"{i} does not support NLP.")

    def persist_result(name: str, _call_kwargs: _CallKwargs, values: _MetricValue):
        if persist_callback is not None:
            persist_callback(name, _call_kwargs, values)

    result = {}
    if call_kwargs is not None:
        # for val in call_kwargs.values():
        #    if not isinstance(val, (List, Dict)):
        #        raise ValueError(
        #            f"Values in call_kwargs must be of type List or Dict, but found {type(val)}"
        #        )
        # Convert regular dict to default dict, so we don't get KeyError.
        call_kwargs_old = call_kwargs.copy()
        call_kwargs = defaultdict(lambda: {})
        call_kwargs.update(call_kwargs_old)
    else:
        call_kwargs = defaultdict(lambda: {})

    global_call_kwargs = {k: v for k, v in call_kwargs.items() if k not in metrics}

    pbar = tqdm(total=len(metrics.keys()), disable=not verbose, desc="Evaluation...")

    with pbar as pbar:
        for metric_name, metric_instance in metrics.items():
            pbar.desc = f"Evaluating {metric_name}"
            metric_call_kwargs = call_kwargs[metric_name]

            if isinstance(metric_call_kwargs, Dict):
                # Metrics kwargs override global ones
                merged_call_kwargs = {**global_call_kwargs, **metric_call_kwargs}
                merged_call_kwargs = add_default_items(
                    merged_call_kwargs, {"a_batch": None}
                )
                scores = metric_instance(model, x_batch, y_batch, **merged_call_kwargs)
                persist_result(metric_name, merged_call_kwargs, scores)
                result[metric_name] = scores
                pbar.update()
                continue

            result[metric_name] = []
            for metric_call_kwarg in metric_call_kwargs:
                # Metrics kwargs override global ones
                merged_call_kwargs = {**global_call_kwargs, **metric_call_kwarg}
                merged_call_kwargs = add_default_items(
                    merged_call_kwargs, {"a_batch": None}
                )
                scores = metric_instance(model, x_batch, y_batch, **merged_call_kwargs)
                persist_result(metric_name, merged_call_kwargs, scores)
                result[metric_name].append(scores)
                pbar.update()

    return result
