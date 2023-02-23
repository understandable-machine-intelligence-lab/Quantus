from __future__ import annotations

from typing import Dict, Optional, List
from tqdm.auto import tqdm
from collections import defaultdict
import gc
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.metrics.batched_metric import BatchedMetric


def evaluate(
    metrics: Dict[str, BatchedMetric],
    model: TextClassifier,
    x_batch: List[str],
    *,
    verbose: bool = True,
    call_kwargs: Optional[Dict[str, Dict[str, ...] | List[Dict[str, ...]]]] = None,
) -> Dict:
    """
    Args:
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
            Internally, `defaultdict` is used, so there is no need to provide kwargs for metrics, which should be evaluated with default ones.

    Returns:

    """
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
        metric_call_kwargs = call_kwargs[metric_name]

        if isinstance(metric_call_kwargs, Dict):
            # Metrics kwargs override global ones
            merged_call_kwargs = {**global_call_kwargs, **metric_call_kwargs}
            scores = metric_instance(model, x_batch, **merged_call_kwargs)
            result[metric_name] = scores
            gc.collect()
            continue

        result[metric_name] = []
        for metric_call_kwarg in metric_call_kwargs:
            # Metrics kwargs override global ones
            merged_call_kwargs = {**global_call_kwargs, **metric_call_kwarg}
            scores = metric_instance(model, x_batch, **merged_call_kwargs)
            result[metric_name].append(scores)
            gc.collect()

    return result
