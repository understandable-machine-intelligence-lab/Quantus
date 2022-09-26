"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""
from typing import Union, Callable, Dict, Optional
import numpy as np

from .helpers import asserts
from .helpers import utils
from .helpers.model_interface import ModelInterface


def evaluate(
    metrics: dict,  # Initialised metrics.
    xai_methods: Union[Dict[str, Callable], Dict[str, np.ndarray], list],
    model: ModelInterface,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch: Union[np.ndarray, None] = None,
    s_batch: Union[np.ndarray, None] = None,
    agg_func: Callable = lambda x: x,
    progress: bool = False,
    explain_func_kwargs: Optional = None,
    **call_kwargs,
) -> Optional[dict]:
    """
    A methods to evaluate metrics given some explanation methods.
    """

    if xai_methods is None:
        print("Define the explanation methods that you want to evaluate.")
        return None

    if metrics is None:
        print(
            "Define the Quantus evaluation metrics that you want to evaluate the explanations against."
        )
        return None

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
                        explain_func_kwargs={"method": method},
                        **call_kwargs,
                    )
                )

    elif isinstance(xai_methods, dict):

        for method, method_func in xai_methods.items():

            results[method] = {}

            if callable(method_func):

                # Asserts.
                asserts.assert_explain_func(explain_func=method_func)

                # Generate explanations.
                a_batch = method_func(
                    model=model,
                    inputs=x_batch,
                    targets=y_batch,
                    **explain_func_kwargs,
                )
                a_batch = utils.expand_attribution_channel(a_batch, x_batch)

                # Asserts.
                asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

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
                        explain_func_kwargs={"method": method},
                        **call_kwargs,
                    )
                )

    return results
