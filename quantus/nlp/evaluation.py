from __future__ import annotations

from typing import Dict, Optional, List
from tqdm.auto import tqdm
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.metrics.batched_metric import BatchedMetric
from quantus.nlp.helpers.types import MetricCallKwargs
from collections import defaultdict


def evaluate(
    metrics: Dict[str, BatchedMetric],
    model: TextClassifier,
    x_batch: List[str],
    verbose: bool = True,
    call_kwargs: Optional[Dict[str, Dict | List[MetricCallKwargs]]] = None,
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
            Dict where keys are metric name, and values are
                1) Dict of kwargs passed to each metric's __call__ method.
                    In this case each metric is evaluated once.
                2) List of `qn.MetricCallKwarg`.
                    In this case each metric is evaluated with each variant of call_kwargs.
                    `qn.MetricCallKwarg.name` is used to identify call_kwargs variant, must be unique for each metric.
            Internally, `defaultdict` is used, so there is no need to provide kwargs for metrics, which should be evaluated with default ones.

    Returns:

    """
    result = {}
    if call_kwargs is not None:
        call_kwargs_old = call_kwargs.copy()
        call_kwargs = defaultdict(lambda: {})
        call_kwargs.update(call_kwargs_old)
    else:
        call_kwargs = defaultdict(lambda: {})

    iter = tqdm(metrics.items(), disable=not verbose, desc="Evaluation...")

    for metric_name, metric_instance in iter:
        metric_call_kwargs = call_kwargs[metric_name]
        if isinstance(metric_call_kwargs, Dict):
            scores = metric_instance(model, x_batch, **metric_call_kwargs)
            result[metric_name] = scores
            continue

        result[metric_name] = {}
        for hyper_params in metric_call_kwargs:
            scores = metric_instance(model, x_batch, **hyper_params.kwargs)
            result[metric_name][hyper_params.name].append(scores)

    return result
