"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import warnings
from typing import Union, Callable, Dict, Optional, Any

import numpy as np
import pandas as pd

from quantus.helpers import asserts
from quantus.helpers import utils
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.explanation_func import explain


def evaluate(
    metrics: Dict,
    xai_methods: Union[Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]],
    model: ModelInterface,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    s_batch: Union[np.ndarray, None] = None,
    agg_func: Callable = lambda x: x,
    explain_func_kwargs: Optional[dict] = None,
    call_kwargs: Union[Dict, Dict[str, Dict]] = None,
    return_as_df: Optional[bool] = None,
    verbose: Optional[bool] = None,
    progress: Optional[bool] = None,
    *args,
    **kwargs,
) -> Optional[dict]:
    """
    Evaluate different explanation methods using specified metrics.

    Parameters
    ----------
    metrics : dict
        A dictionary of initialized evaluation metrics. See quantus.AVAILABLE_METRICS.
        Example: {'Robustness': quantus.MaxSensitivity(), 'Faithfulness': quantus.PixelFlipping()}

    xai_methods : dict
        A dictionary specifying the explanation methods to evaluate, which can be structured in three ways:

        1) Dict[str, Dict] for built-in Quantus methods (using quantus.explain):

            Example:
            xai_methods = {
                'IntegratedGradients': {
                    'n_steps': 10,
                    'xai_lib': 'captum'
                },
                'Saliency': {
                    'xai_lib': 'captum'
                }
            }

            - See quantus.AVAILABLE_XAI_METHODS_CAPTUM for supported captum methods.
            - See quantus.AVAILABLE_XAI_METHODS_TF for supported tensorflow methods.
            - See https://github.com/chr5tphr/zennit for supported zennit methods.
            - Read more about the explanation function arguments here:
              <https://quantus.readthedocs.io/en/latest/docs_api/quantus.functions.explanation_func.html#quantus.functions.explanation_func.explain>

        2) Dict[str, Callable] for custom methods:

            Example:
            xai_methods = {
                'custom_own_xai_method': custom_explain_function
            }
            or
            ai_methods = {"InputXGradient": {
        "explain_func": quantus.explain,
        "explain_func_kwargs": {},
    }}

            - Here, you can provide your own callable that mirrors the input and outputs of the quantus.explain() method.

        3) Dict[str, np.ndarray] for pre-calculated attributions:

            Example:
            xai_methods = {
                'LIME': precomputed_numpy_lime_attributions,
                'GradientShap': precomputed_numpy_shap_attributions
            }

            - Note that some Quantus metrics, e.g., quantus.MaxSensitivity() within the robustness
            category, includes "re-explaning" the input and output pair as a part of the evaluation metric logic.
            If you include such metrics in the quantus.evaluate(), this option will not be possible.

        It is also possible to pass a combination of the above.

        >>> xai_methods = {
        >>>     'IntegratedGradients': {
        >>>         'n_steps': 10,
        >>>         'xai_lib': 'captum'
        >>>     },
        >>>     'Saliency': {
        >>>         'xai_lib': 'captum'
        >>>     },
        >>>     'custom_own_xai_method': custom_explain_function,
        >>>     'LIME': precomputed_numpy_lime_attributions,
        >>>     'GradientShap': precomputed_numpy_shap_attributions
        >>> }

    model: Union[torch.nn.Module, tf.keras.Model]
        A torch or tensorflow model that is subject to explanation.

    x_batch: np.ndarray
        A np.ndarray containing the input data to be explained.

    y_batch: np.ndarray
        A np.ndarray containing the output labels corresponding to x_batch.

    s_batch: np.ndarray, optional
        A np.ndarray containing segmentation masks that match the input.

    agg_func: Callable
        Indicates how to aggregate scores, e.g., pass np.mean.

    explain_func_kwargs: dict, optional
        Keyword arguments to be passed to explain_func on call. Pass None if using Dict[str, Dict] type for xai_methods.

    call_kwargs: Dict[str, Dict]
        Keyword arguments for the call of the metrics. Keys are names for argument sets, and values are argument dictionaries.

    verbose: optional, bool
        Indicates whether to print evaluation progress.

    progress: optional, bool
        Deprecated. Indicates whether to print evaluation progress. Use verbose instead.

    return_as_df: optional, bool
        Indicates whether to return the results as a pd.DataFrame. Only works if call_kwargs is not passed.

    args: optional
        Deprecated arguments for the call.

    kwargs: optional
        Deprecated keyword arguments for the call of the metrics.

    Returns
    -------
    results: dict
        A dictionary with the evaluation results.
    """
    warn.check_kwargs(kwargs)

    if xai_methods is None:
        print("Define the explanation methods that you want to evaluate.")
        return None

    if metrics is None:
        print(
            "Define the Quantus evaluation metrics that you want to evaluate the explanations against."
        )
        return None

    if call_kwargs is None:
        call_kwargs = {"call_kwargs_empty": {}}

    elif not isinstance(call_kwargs, Dict):
        raise TypeError("call_kwargs type should be of Dict[str, Dict] (if not None).")

    if progress is not None:
        warnings.warn(
            "'progress' parameter is deprecated and will be removed in future versions. "
            "Please use 'verbose' instead. ",
            DeprecationWarning,
        )
        verbose = progress  # Use the value of 'progress' for 'verbose'

    results: Dict[str, dict] = {}
    explain_funcs: Dict[str, Callable] = {}

    if not isinstance(xai_methods, dict):
        "Make sure that 'xai_methods' is of type: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."

    for method, value in xai_methods.items():

        results[method] = {}

        if callable(value):

            explain_funcs[method] = value
            explain_func = value
            assert (
                explain_func_kwargs is not None
            ), "Pass explain_func_kwargs as a separate argument (dictionary)."

            # Asserts.
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **{**explain_func_kwargs, **{"method": method}},
            )
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)

            # Asserts.
            asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        elif isinstance(value, Dict):

            if explain_func_kwargs is not None:

                warnings.warn(
                    "Passed explain_func_kwargs will be ignored when passing type Dict[str, Dict] as xai_methods."
                    "Read the docstring section on xai_methods (part 1) for more information."
                )

            explain_func_kwargs = value
            explain_funcs[method] = explain

            # Generate explanations.
            a_batch = explain(
                model=model, inputs=x_batch, targets=y_batch, **explain_func_kwargs
            )
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)

            # Asserts.
            asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        elif isinstance(value, np.ndarray):
            explain_funcs[method] = explain
            a_batch = value

        else:

            raise TypeError(
                f"Error: Unsupported xai_methods type for method '{method}'."
            )

        if explain_func_kwargs is None:
            explain_func_kwargs = {}

        for (metric, metric_func) in metrics.items():

            results[method][metric] = {}

            for (call_kwarg_str, call_kwarg) in call_kwargs.items():

                if verbose:
                    print(
                        f"Evaluating {method} explanations on {metric} metric with "
                        f"call parameters: {call_kwarg}..."
                    )

                try:
                    scores = metric_func(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        s_batch=s_batch,
                        explain_func=explain_funcs[method],
                        explain_func_kwargs={
                            **explain_func_kwargs,
                            **{"method": method},
                        },
                        **call_kwarg,
                        **kwargs,
                    )

                    results[method][metric][call_kwarg_str] = agg_func(scores)

                except Exception as e:
                    print(
                        f"\nError {e}.\nFailed to evaluate {method} explanations on {metric} metric with "
                        f"call parameters: {call_kwarg_str}. \nPlease be aware that passing the explanation "
                        f"as a Numpy array may not be possible if the metric logic necessitates re-explaining"
                        f" the input. This requirement is common in metrics related to robustness and randomisation. "
                        f"Please review the documentation for the specific metric to verify this requirement."
                    )

    results_ordered: Dict[str, Any] = {}  # type: ignore

    if len(call_kwargs) == 1:

        # Clean up the results if there is only one call_kwarg.
        for method, value in xai_methods.items():
            results_ordered[method] = {}
            for (metric, metric_func) in metrics.items():
                for (call_kwarg_str, call_kwarg) in call_kwargs.items():
                    results_ordered[method][metric] = results[method][metric][
                        call_kwarg_str
                    ]

    if return_as_df:
        if len(call_kwargs) > 1:
            print(
                "Returning the results as a pd.DataFrame is only possible if the 'call_kwargs' "
                "is None or is a dictionary of length of 1 (i.e., triggers one evaluation run)."
            )
            return results
        else:
            return pd.DataFrame.from_dict(results_ordered)

    if len(call_kwargs) > 1:
        return results

    return results_ordered
