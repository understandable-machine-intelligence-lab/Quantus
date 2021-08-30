from typing import Union, Callable, Dict
import numpy as np
from .metrics import *
from .helpers.constants import *


def evaluate(
    evaluation_metrics: dict,
    explanation_methods: Union[Dict[str, Callable], Dict[str, np.ndarray], list],
    model: torch.nn,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch: Union[np.ndarray, None] = None,
    agg_func: Callable = lambda x: x,
    **kwargs,
) -> dict:
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
                    a_batch = method_func(
                        model=model,
                        inputs=x_batch,
                        targets=y_batch,
                        **{**kwargs, **{"explanation_func": method}},
                    )

                else:

                    if not isinstance(method_func, np.ndarray):
                        raise TypeError(
                            "Explanations must be of type np.ndarray or a Callable function that outputs np.nparray."
                        )

                results[metric][method] = agg_func(
                    metric_func(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        **{**kwargs, **{"explanation_func": method}},
                    )
                )

        elif isinstance(explanation_methods, list):

            for method in explanation_methods:
                results[metric][method] = agg_func(
                    metric_func(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        **{**kwargs, **{"explanation_func": method}},
                    )
                )

    return results


def available_categories():
    for c in AVAILABLE_METRICS.keys():
        print(f"Category {c}")


def available_metrics():
    for c in AVAILABLE_METRICS.keys():
        for m in AVAILABLE_METRICS[c].keys():
            print(f"\tMetric {m} ({c} category)")


def available_methods():
    for xai in AVAILABLE_XAI_METHODS.keys():
        print(f"Method {xai}")


def available_perturbation_functions():
    for func in AVAILABLE_PERTURBATION_FUNCTIONS.keys():
        print(f"Perturbation function - {func}")


def available_similarity_functions():
    for func in AVAILABLE_SIMILARITY_FUNCTIONS.keys():
        print(f"Similarity function - {func}")


def available_localization_functions():
    for func in AVAILABLE_LOCALIZATION_FUNCTIONS.keys():
        print(f"Localization function - {func}")
