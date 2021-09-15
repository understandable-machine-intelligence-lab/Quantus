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
    s_batch: np.ndarray,
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
    s_batch
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

                a_batch = method_func

                if callable(method_func):

                    # Asserts.
                    explain_func = kwargs.get("explain_func", Callable)
                    assert_explain_func(explain_func=explain_func)

                    # Generate explanations.
                    a_batch = explain_func(
                        model=model,
                        inputs=x_batch,
                        targets=y_batch,
                        **kwargs,
                    )

                    # Asserts.
                    assert_attributions(a_batch=a_batch, x_batch=x_batch)

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
                        s_batch=s_batch,
                        **{**kwargs, **{"explain_func": method}},
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
                        s_batch=s_batch,
                        **{**kwargs, **{"explain_func": method}},
                    )
                )

    return results


def available_categories() -> list:
    return [c for c in AVAILABLE_METRICS.keys()]


def available_metrics() -> dict:
    return {c : list(metrics.keys()) for c, metrics in AVAILABLE_METRICS.items()}


def available_methods() -> list:
    return [c for c in AVAILABLE_XAI_METHODS.keys()]


def available_perturbation_functions() -> list:
    return [c for c in AVAILABLE_PERTURBATION_FUNCTIONS.keys()]


def available_similarity_functions() -> list:
    return [c for c in AVAILABLE_SIMILARITY_FUNCTIONS.keys()]


def available_localization_functions() -> list:
    return [c for c in AVAILABLE_LOCALIZATION_FUNCTIONS.keys()]
