"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""
from typing import Union, Callable, Dict
import numpy as np
from .metrics import *
from .helpers.constants import *
from .helpers.model_interface import ModelInterface


def evaluate(
    metrics: dict,
    xai_methods: Union[Dict[str, Callable], Dict[str, np.ndarray], list],
    model: ModelInterface,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch: Union[np.ndarray, None] = None,
    s_batch: Union[np.ndarray, None] = None,
    agg_func: Callable = lambda x: x,
    progress: bool = False,
    *args,
    **kwargs,
) -> dict:
    """
    A methods to evaluate metrics given some explanation methods.

    Parameters
    ----------
    metrics
    xai_methods
    model
    x_batch
    y_batch
    s_batch
    agg_func
    kwargs

    Returns
    -------

    """

    if xai_methods is None:
        print("Define the explanation methods that you want to evaluate.")

    if metrics is None:
        print(
            "Define the Quantus evaluation metrics that you want to evaluate the explanations against."
        )

    results = {}

    if isinstance(xai_methods, list):

        assert a_batch is not None, (
            "If 'explanation_methods' is a list of methods as strings, "
            "then a_batch arguments should provide the necessary attributions corresponding "
            "to each input."
        )

        for method in xai_methods:

            results[method] = {}

            for metric, metric_func in metrics.items():

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

    elif isinstance(xai_methods, dict):

        for method, method_func in xai_methods.items():

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
                a_batch = utils.expand_attribution_channel(a_batch, x_batch)

                # Asserts.
                assert_attributions(a_batch=a_batch, x_batch=x_batch)

            elif isinstance(method_func, np.ndarray):

                a_batch = method_func

            else:

                if not isinstance(method_func, np.ndarray):
                    raise TypeError(
                        "Explanations must be of type np.ndarray or a Callable function that outputs np.nparray."
                    )

            for metric, metric_func in metrics.items():

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
