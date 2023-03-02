from __future__ import annotations

from typing import Dict, Optional, List, Callable, Union, Any

import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
import gc
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.metrics.batched_metric import BatchedMetric

_MetricValue = Union[np.ndarray, float, Dict[str, np.ndarray]]
_CallKwargs = Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]
_PersistFn = Callable[[str, _CallKwargs, _MetricValue], None]


def evaluate(
    metrics: Dict[str, BatchedMetric],
    model: TextClassifier,
    x_batch: List[str],
    *,
    verbose: bool = True,
    run_gc: bool = False,
    call_kwargs: Optional[_CallKwargs] = None,
    persist_callback: Optional[_PersistFn] = None,
) -> Dict[str, _MetricValue | List[_MetricValue]]:
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
        - keys are metrics' names, and values are
        - Kwargs passed to each metrics' __call__ method. In this case each metric is evaluated once.
        - List of dicts. In this case each metric is evaluated with each entry of list of call_kwargs.
        - All key, values, where are not in metric names, kwargs are treated as global and passed to each metric.
        Metrics-specific kwargs will override global ones.
        Internally, `defaultdict` is used, so there is no need to provide kwargs for metrics,
            which should be evaluated with default ones.
    run_gc:
        Indicates, if garbage collection should be manually run every time after metric is evaluated.
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

    def collect_garbage():
        if run_gc:
            gc.collect()

    def persist_result(name: str, _call_kwargs: _CallKwargs, values: _MetricValue):
        if persist_callback is not None:
            persist_callback(name, _call_kwargs, values)

    result = {}
    if call_kwargs is not None:
        for val in call_kwargs.values():
            if not isinstance(val, (List, Dict)):
                raise ValueError(
                    f"Values in call_kwargs must be of type List or Dict, but found {type(val)}"
                )
        # Convert regular dict to default dict, so we don't get KeyError.
        call_kwargs_old = call_kwargs.copy()
        call_kwargs = defaultdict(lambda: {})
        call_kwargs.update(call_kwargs_old)
    else:
        call_kwargs = defaultdict(lambda: {})

    global_call_kwargs = {k: v for k, v in call_kwargs.items() if k not in metrics}

    iterator = tqdm(metrics.items(), disable=not verbose, desc="Evaluation...")

    for metric_name, metric_instance in iterator:
        iterator.desc = f"Evaluating {metric_name}"
        metric_call_kwargs = call_kwargs[metric_name]

        if isinstance(metric_call_kwargs, Dict):
            # Metrics kwargs override global ones
            merged_call_kwargs = {**global_call_kwargs, **metric_call_kwargs}
            scores = metric_instance(model, x_batch, **merged_call_kwargs)
            collect_garbage()
            persist_result(metric_name, merged_call_kwargs, scores)
            result[metric_name] = scores
            continue

        result[metric_name] = []
        for metric_call_kwarg in metric_call_kwargs:
            # Metrics kwargs override global ones
            merged_call_kwargs = {**global_call_kwargs, **metric_call_kwarg}
            scores = metric_instance(model, x_batch, **merged_call_kwargs)
            collect_garbage()
            persist_result(metric_name, merged_call_kwargs, scores)
            result[metric_name].append(scores)
            collect_garbage()

    return result
