"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import warnings
from typing import Union, Callable, Dict, Optional, List

import numpy as np

from quantus.helpers import asserts
from quantus.helpers import utils
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.enums import ScoreDirection, EvaluationCategory
from quantus.functions.explanation_func import explain
from quantus.functions import postprocess_func


def evaluate(
    metrics: Dict,
    xai_methods: Union[Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]],
    model: ModelInterface,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    s_batch: Union[np.ndarray, None] = None,
    agg_func: Callable = lambda x: x,
    progress: bool = False,
    explain_func_kwargs: Optional[dict] = None,
    call_kwargs: Union[Dict, Dict[str, Dict]] = None,
    return_skill_score: bool = False,
    **kwargs,
) -> Optional[dict]:
    """
    A method to evaluate some explanation methods given some metrics.

    Parameters
    ----------
    metrics: dict
        A dictionary with intialised metrics.
    xai_methods: dict, list
        Pass the different explanation methods as:
        1) Dict[str, np.ndarray] where values are pre-calculcated attributions, or
        2) Dict[str, Dict] where the keys are the name of the Quantus build-in explanation methods,
        and the values are the explain function keyword arguments as a dictionary, or
        3) Dict[str, Callable] where the keys are the name of explanation methods,
        and the values a callable explanation function.
    model: torch.nn.Module, tf.keras.Model
        A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
    x_batch: np.ndarray
        A np.ndarray which contains the input data that are explained.
    y_batch: np.ndarray
        A np.ndarray which contains the output labels that are explained.
    s_batch: np.ndarray, optional
        A np.ndarray which contains segmentation masks that matches the input.
    agg_func: callable
        Indicates how to aggregates scores e.g., pass np.mean.
    progress: boolean
        Indicates if progress should be printed to std, or not.
    explain_func_kwargs: dict, optional
        Keyword arguments to be passed to explain_func on call. Pass None if using Dict[str, Dict] type for xai_methods.
    call_kwargs: Dict[str, Dict]
        Keyword arguments for the call of the metrics, keys are names for arg set and values are argument dictionaries.
    return_skill_score: boolean
        Indicates if skill score is to be returned.
    kwargs: optional
        Deprecated keyword arguments for the call of the metrics.
    Returns
    -------
    results: dict
        A dictionary with the results.
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
        call_kwargs = {}
    elif not isinstance(call_kwargs, Dict):
        raise TypeError("xai_methods type is not Dict[str, Dict].")

    results: Dict[str, dict] = {}
    explain_funcs: Dict[str, Callable] = {}

    if not isinstance(xai_methods, dict):
        "xai_methods type is not in: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."

    for method, value in xai_methods.items():

        results[method] = {}

        if callable(value):

            explain_funcs[method] = value
            explain_func = value

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
                    "Pass explanation arguments as dictionary values."
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
                "xai_methods type is not in: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."
            )

        if explain_func_kwargs is None:
            explain_func_kwargs = {}

        for (metric, metric_func) in metrics.items():

            results[method][metric] = {}

            for (call_kwarg_str, call_kwarg) in call_kwargs.items():

                if progress:
                    print(
                        f"Evaluating {method} explanations on {metric} metric on set of call parameters {call_kwarg_str}..."
                    )

                if return_skill_score:

                    # Remove the aggregate function.
                    agg_func_old = agg_func
                    agg_func = np.array

                    # Make sure to return one value per explanation.
                    if metric_func.return_aggregate:
                        metric_func.return_aggregate = False
                        print(
                            "'return_aggregate' was set to True. To calculate skill score, it is now set to False."
                        )

                    # Specific case for MPT, return the average correlation per sample.
                    if (
                        hasattr(metric_func, "return_sample_correlation")
                        and not metric_func.return_sample_correlation
                    ):
                        metric_func.return_sample_correlation = True
                        print(
                            "'return_sample_correlation' was set to False. To calculate skill score, it is now set to True."
                        )

                results[method][metric][call_kwarg_str] = agg_func(
                    metric_func(
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
                )

                if return_skill_score:

                    # Get the worst-case explanation.
                    method_control = "Control Var. Random Uniform"
                    explain_funcs[method_control] = explain

                    # Special case, randomisation category.
                    if metric_func.evaluation_category == EvaluationCategory.RANDOMISATION:

                        # Generate an explanations and score them constantly.
                        a_batch = explain_funcs[method_control](
                            model=model,
                            inputs=x_batch,
                            targets=y_batch,
                            **{**explain_func_kwargs,
                               **{"method": method_control},
                               }
                        )

                        # Measure similarity against the same explanation.
                        scores_reference = [metric_func.similarity_func(a.flatten(), a.flatten()) for a in a_batch]

                    else:

                        scores_reference = agg_func(
                            metric_func(
                                model=model,
                                x_batch=x_batch,
                                y_batch=y_batch,
                                a_batch=a_batch,
                                s_batch=s_batch,
                                explain_func=explain_funcs[method_control],
                                explain_func_kwargs={
                                    **explain_func_kwargs,
                                    **{"method": method_control},
                                },
                                **call_kwarg,
                                **kwargs,
                            )
                        )

                    print(f"\n{metric} ({metric_func.score_direction.name} is better)\n  Baseline: "
                          f"{scores_reference}\n  {method}: {results[method][metric][call_kwarg_str]}")

                    # Compute the skill score.
                    skill_score = postprocess_func.explanation_skill_score(
                        y_scores=results[method][metric][call_kwarg_str],
                        y_refs=scores_reference,
                        score_direction=metric_func.score_direction,
                        agg_func=agg_func_old,
                    )

                    """
                    # Compute the skill score.
                    xai_score = postprocess_func.xai_skill_score(
                        y_scores=results[method][metric][call_kwarg_str],
                        y_refs=scores_reference,
                        score_direction=metric_func.score_direction,
                        #agg_func=agg_func_old,
                    )
                    """

                    print("Skill:", skill_score)
                    #print("Skill *:", xai_score)

                    results[method][metric][call_kwarg_str] = skill_score

    return results
