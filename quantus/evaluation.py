"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import Union, Callable, Dict, Optional, List
import numpy as np

from .helpers import asserts
from .helpers import utils
from .helpers.model_interface import ModelInterface


def evaluate(
    metrics: Dict,
    xai_methods: Union[Dict[str, Callable], Dict[str, np.ndarray], List[str]],
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
    A method to evaluate some explanation methods given some metrics.

    Parameters
    ----------
    metrics: dict
        A dictionary with intialised metrics.
    xai_methods: dict, list
        Pass the different explanation methods, either: as a List[str] using the included explanation methods
        in Quantus. Or as a dictionary with where the keys are the name of the explanation methods and the values
        are the explanations (np.array). Alternatively, pass an explanation function to compute on-the-fly
        instead of passing pre-computed attributions as the dictionary values.
    model: Union[torch.nn.Module, tf.keras.Model]
        A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
    x_batch: np.ndarray
        A np.ndarray which contains the input data that are explained.
    y_batch: np.ndarray
        A np.ndarray which contains the output labels that are explained.
    a_batch: np.ndarray, optional
        A np.ndarray which contains pre-computed attributions i.e., explanations.
    s_batch: np.ndarray, optional
        A np.ndarray which contains segmentation masks that matches the input.
    agg_func: callable
        Indicates how to aggregates scores e.g., pass np.mean.
    progress: boolean
        Indicates if progress should be printed to std, or not.
    explain_func_kwargs: dict, optional
        Keyword arguments to be passed to explain_func on call.
    call_kwargs: optional
        Keyword arguments for the call of the metrics.

    Returns
    -------
    results: dict
        A dictionary with the results.
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
                        explain_func_kwargs={
                            **explain_func_kwargs,
                            **{"method": method},
                        },
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
                    **{**explain_func_kwargs, **{"method": method}},
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
                        explain_func_kwargs={
                            **explain_func_kwargs,
                            **{"method": method},
                        },
                        **call_kwargs,
                    )
                )

    return results
