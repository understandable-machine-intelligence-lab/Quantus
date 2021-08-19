from typing import Union, Callable, Dict
import numpy as np
from .metrics import *
from .helpers.constants import DEFAULT_XAI_METHODS, DEFAULT_METRICS


def evaluate(evaluation_metrics: dict,
             explanation_methods: Union[Dict[str, Callable], Dict[str, np.ndarray], list],
             model: torch.nn,
             x_batch: np.ndarray,
             y_batch: np.ndarray,
             a_batch: Union[np.ndarray, None] = None,
             agg_func: Callable = lambda x: x,
             **kwargs) -> dict:
    """
    A methods to evaluate metrics given some explanation methods.

    Parameters
    ----------
    evaluation_metrics
    explanation_methods
    model
    x_batch
    y_batch
    a_batch
    agg_func
    kwargs

    Returns
    -------

    """

    if explanation_methods is None:
        explanation_methods = DEFAULT_XAI_METHODS

    if evaluation_metrics is None:
        evaluation_metrics = DEFAULT_METRICS

    results = {}

    for metric, metric_func in evaluation_metrics.items():

        results[metric] = {}

        if isinstance(explanation_methods, dict):

            for method, method_func in explanation_methods.items():

                if callable(method_func):

                    # TODO. Write placeholder function for explanations.
                    a_batch = method_func(model=model,
                                          inputs=x_batch,
                                          targets=y_batch,
                                          **{**kwargs, **{"explanation_func": method}})

                else:

                    if not isinstance(method_func, np.ndarray):
                        raise TypeError("Explanations must be of type np.ndarray.")

                results[metric][method] = agg_func(metric_func(model=model,
                                                               x_batch=x_batch,
                                                               y_batch=y_batch,
                                                               a_batch=a_batch,
                                                               **{**kwargs, **{"explanation_func": method}}))

        elif isinstance(explanation_methods, list):

            for method in explanation_methods:
                results[metric][method] = agg_func(metric_func(model=model,
                                                               x_batch=x_batch,
                                                               y_batch=y_batch,
                                                               a_batch=a_batch,
                                                               **{**kwargs, **{"explanation_func": method}}))

    return results