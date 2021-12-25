from typing import Union, Callable, Dict
import torch
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
    s_batch: Union[np.ndarray, None] = None,
    agg_func: Callable = lambda x: x,
    progress: bool = False,
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
        print("Define the explanation methods that you want to evaluate.")

    if evaluation_metrics is None:
        print(
            "Define the Quantus evaluation metrics that you want to evaluate the explanations against."
        )

    results = {}

    if isinstance(explanation_methods, list):

        assert a_batch is not None, (
            "If 'explanation_methods' is a list of methods as strings, "
            "then a_batch arguments should provide the necessary attributions corresponding "
            "to each input."
        )

        for method in explanation_methods:

            results[method] = {}

            for metric, metric_func in evaluation_metrics.items():

                if progress:
                    print(f"Evaluating {method} explanations on {metric} metric...")

                results[method][metric] = agg_func(
                    metric_func(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        s_batch=s_batch,
                        **{**kwargs, **{"method": method}},
                    )
                )

    elif isinstance(explanation_methods, dict):

        for method, method_func in explanation_methods.items():

            results[method] = {}

            if callable(method_func):

                # Asserts.
                assert_explain_func(explain_func=method_func)

                # Generate explanations.
                a_batch = method_func(
                    model=model,
                    inputs=x_batch,
                    targets=y_batch,
                    **kwargs,
                )

                # Asserts.
                assert_attributions(a_batch=a_batch, x_batch=x_batch)

            elif isinstance(method_func, np.ndarray):

                a_batch = method_func

            else:

                if not isinstance(method_func, np.ndarray):
                    raise TypeError(
                        "Explanations must be of type np.ndarray or a Callable function that outputs np.nparray."
                    )

            for metric, metric_func in evaluation_metrics.items():

                if progress:
                    print(f"Evaluating {method} explanations on {metric} metric...")

                results[method][metric] = agg_func(
                    metric_func(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        s_batch=s_batch,
                        **{**kwargs, **{"method": method}},
                    )
                )

    return results


def available_categories() -> list:
    return [c for c in AVAILABLE_METRICS.keys()]


def available_metrics() -> dict:
    return {c: list(metrics.keys()) for c, metrics in AVAILABLE_METRICS.items()}


def available_methods() -> list:
    return [c for c in AVAILABLE_XAI_METHODS.keys()]


def available_perturbation_functions() -> list:
    return [c for c in AVAILABLE_PERTURBATION_FUNCTIONS.keys()]


def available_similarity_functions() -> list:
    return [c for c in AVAILABLE_SIMILARITY_FUNCTIONS.keys()]


def available_normalisation_functions() -> list:
    return [c for c in AVAILABLE_NORMALISATION_FUNCTIONS.keys()]
