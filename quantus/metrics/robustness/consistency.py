"""This module contains the implementation of the Consistency metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base import Metric
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.discretise_func import top_n_sign


class Consistency(Metric):
    """

    The (global) consistency metric measures the expected local consistency. Local consistency measures the probability
    of the prediction label for a given datapoint coinciding with the prediction labels of other data points that
    the same explanation is being attributed to. For example, if the explanation of a given image is "contains zebra",
    the local consistency metric measures the probability a different image that the explanation "contains zebra" is
    being attributed to having the same prediction label.

    References:
         1) Sanjoy Dasgupta, Nave Frost, and Michal Moshkovitz. "Framework for Evaluating Faithfulness of Local
            Explanations." arXiv preprint arXiv:2202.00734 (2022).

    Assumptions:
        - A used-defined discreization function is used to discretize continuous explanation spaces.
    """

    @asserts.attributes_check
    def __init__(
        self,
        discretise_func: Optional[Callable] = None,
        abs: bool = True,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        discretise_func (callable): Discretisation function applied to explantions.
            If None, the default value is used, default=top_n_sign.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=gaussian_noise.
        perturb_std (float): The amount of noise added, default=0.1.
        perturb_mean (float): The mean of noise added, default=0.0.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate (boolean): Indicates if an aggregated score should be computed over all instances.
        aggregate_func (callable): Callable that aggregates the scores given an evaluation call.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        if discretise_func is None:
            discretise_func = top_n_sign
        self.discretise_func = discretise_func
        self.y_pred_classes = None

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "Function for discretisation of the explanation space 'discretise_func' (return hash value of"
                    "an np.array used for comparison)."
                ),
                citation=(
                    "Sanjoy Dasgupta, Nave Frost, and Michal Moshkovitz. 'Framework for Evaluating Faithfulness of "
                    "Explanations.' arXiv preprint arXiv:2202.00734 (2022)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        custom_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict[str, Any]] = None,
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
        softmax: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ) -> List[float]:
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=custom_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

    def evaluate_instance(
        self,
        i: int,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        c: Any,
        p: Any,
    ) -> float:

        # Unpack custom preprocess.
        a_label = c

        # Metric logic.
        pred_a = self.y_pred_classes[i]
        same_a = np.argwhere(a == a_label).flatten()
        diff_a = same_a[same_a != i]
        pred_same_a = self.y_pred_classes[diff_a]

        if len(same_a) == 0:
            return 0
        return np.sum(pred_same_a == pred_a) / len(diff_a)

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> Tuple[
        ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any
    ]:

        custom_preprocess_batch = [None for _ in x_batch]

        # Preprocessing.
        a_batch_flat = a_batch.reshape(a_batch.shape[0], -1)
        a_labels = np.array(list(map(self.discretise_func, a_batch_flat)))

        x_input = model.shape_input(
            x_batch, x_batch[0].shape, channel_first=True, batched=True
        )
        self.y_pred_classes = np.argmax(model.predict(x_input), axis=1).flatten()

        custom_preprocess_batch = a_labels

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
