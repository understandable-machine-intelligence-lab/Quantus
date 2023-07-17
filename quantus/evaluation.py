"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import warnings
from typing import Union, Callable, Dict, Optional, List

import pandas as pd
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
    return_as_df: bool = False,
    progress: bool = False,
    explain_func_kwargs: Optional[dict] = None,
    call_kwargs: Optional[Union[Dict, Dict[str, Dict]]] = None,
    return_skill_score: bool = False,
    agg_func_skill_score: Optional[Callable] = lambda x: x,
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
    return_as_df: boolean
        Indicates if a pd.DataFrame should be returned.
    progress: boolean
        Indicates if progress should be printed to std, or not.
    explain_func_kwargs: dict, optional
        Keyword arguments to be passed to explain_func on call. Pass None if using Dict[str, Dict] type for xai_methods.
    call_kwargs: Dict[str, Dict]
        Keyword arguments for the call of the metrics, keys are names for arg set and values are argument dictionaries.
    return_skill_score: boolean
        Indicates if skill score is to be returned.
    agg_func_skill_score: callable, None
        Indicates how to aggregates skill scores e.g., pass np.nanmean, optional.
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
        call_kwargs = {'call_kwargs_empty': {}}

    elif not isinstance(call_kwargs, Dict):
        raise TypeError("xai_methods type is not Dict[str, Dict].")

    evaluation_results: Dict[str, dict] = {}
    explain_funcs: Dict[str, Callable] = {}

    if not isinstance(xai_methods, dict):
        "xai_methods type is not in: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."

    evaluation_scores_raw = {}
    evaluation_scores_refs = {}

    for xai_method, xai_method_value in xai_methods.items():

        evaluation_results[xai_method] = {}
        evaluation_scores_raw[xai_method] = {}

        if callable(xai_method_value):

            explain_funcs[xai_method] = xai_method_value
            explain_func = xai_method_value

            # Asserts.
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **{**explain_func_kwargs, **{"method": xai_method}},
            )
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)

            # Asserts.
            asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        elif isinstance(xai_method_value, Dict):

            if explain_func_kwargs:
                warnings.warn(
                    "Passed 'explain_func_kwargs' will be ignored when passing type Dict[str, Dict] as xai_methods. "
                    "Pass explanation arguments as dictionary values for 'xai_methods'."
                )

            explain_func_kwargs = xai_method_value
            explain_funcs[xai_method] = explain

            # Generate explanations.
            a_batch = explain(
                model=model, inputs=x_batch, targets=y_batch, **explain_func_kwargs
            )
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)

            # Asserts.
            asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        elif isinstance(xai_method_value, np.ndarray):
            explain_funcs[xai_method] = explain
            a_batch = xai_method_value

        else:
            raise TypeError(
                "xai_methods type is not in: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."
            )

        if explain_func_kwargs is None:
            explain_func_kwargs = {}

        if return_skill_score:

            # Remove the aggregate function.
            agg_func_args = agg_func
            agg_func = np.array

            # Get the worst-case explanation.
            method_control = "Control Var. Random Uniform"
            explain_funcs[method_control] = explain

            # Generate an explanations and score them constantly.
            a_batch_control = explain_funcs[method_control](
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **{**explain_func_kwargs,
                **{"method": method_control},
                }
            )

        for (metric, metric_func) in metrics.items():

            evaluation_results[xai_method][metric] = {}
            evaluation_scores_raw[xai_method][metric] = {}
            evaluation_scores_refs[metric] = {}

            for (call_kwarg_str, call_kwarg) in call_kwargs.items():

                # TODO. Fix evaluation_scores_raw keys if len(call_kwargs) > 1:
                # TODO. Fix evaluation_scores_refs keys if len(call_kwargs) > 1:

                if progress:
                    print(
                        f"Evaluating {xai_method} explanations on {metric} metric on set of call parameters {call_kwarg_str}..."
                    )

                # Compute evaluation scores.
                evaluation_scores = agg_func(
                    metric_func(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        s_batch=s_batch,
                        explain_func=explain_funcs[xai_method],
                        explain_func_kwargs={
                            **explain_func_kwargs,
                            **{"method": xai_method},
                        },
                        **call_kwarg,
                        **kwargs,
                    )
                )

                evaluation_scores_raw[xai_method][metric]["evaluation_scores"] = evaluation_scores

                if return_skill_score:

                    # Preprocessing steps.

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

                    if "reference_scores" not in evaluation_scores_refs[metric]:
                        # Special case, randomisation category.
                        if metric_func.evaluation_category == EvaluationCategory.RANDOMISATION:

                            # Measure similarity against the same explanation.
                            reference_scores = [metric_func.similarity_func(a_batch_control.flatten(), a_batch_control.flatten()) for a in a_batch]

                        else:

                            # Compute reference scores.
                            reference_scores = agg_func(
                                metric_func(
                                    model=model,
                                    x_batch=x_batch,
                                    y_batch=y_batch,
                                    a_batch=a_batch_control,
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


                    print(f"\n{metric} ({metric_func.score_direction.name} is better)\n"
                          f" Baseline: {reference_scores}\n  {xai_method}: {evaluation_scores}")

                    if agg_func_skill_score is None:
                        agg_func_skill_score = np.nanmean

                    # Compute the skill score.
                    skill_scores = postprocess_func.explanation_skill_score(
                        y_scores=evaluation_scores,
                        y_refs=reference_scores,
                        score_direction=metric_func.score_direction,
                        agg_func=agg_func_skill_score,
                    )
                    print("Skill:", skill_scores)

                    evaluation_scores_refs[metric]["reference_scores"] = reference_scores
                    evaluation_scores_raw[xai_method][metric]["skill_scores"] = skill_scores

                    evaluation_scores = skill_scores

                # If there is only one set of call_kwargs we drop the key.
                if len(call_kwargs) > 1:
                    evaluation_results[xai_method][metric][call_kwarg_str] = evaluation_scores
                else:
                    evaluation_results[xai_method][metric] = evaluation_scores

    # Convert the result to pd.DataFrame.
    if return_as_df:
        if len(call_kwargs) > 1:
            print("Conversion from dict to pd.DataFrame of quantus.evaluate results is only possible if no dict is passed to 'call_kwargs' argument.")
            return evaluation_results, evaluation_scores_raw, evaluation_scores_refs
        else:
            try:
                return pd.DataFrame(evaluation_results), evaluation_scores_raw, evaluation_scores_refs
            except:
                print("Tried to convert the quantus.evaluate results to pd.DataFrame but failed. Try to set 'return_as_df' to False.")
                return evaluation_results, evaluation_scores_raw, evaluation_scores_refs

    return evaluation_results, evaluation_scores_raw, evaluation_scores_refs
