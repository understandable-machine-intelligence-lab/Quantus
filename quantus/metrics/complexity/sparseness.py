"""This module contains the implementation of the Sparseness metric."""

from typing import Any, Callable, Dict, List, Optional
import numpy as np

from ..base import Metric
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative


class Sparseness(Metric):
    """
    Implementation of Sparseness metric by Chalasani et al., 2020.

    Sparseness is quantified using the Gini Index applied to the vector of the absolute values of attributions. The
    test asks that features that are truly predictive of the output F(x) should have significant contributions, and
    similarly, that irrelevant (or weakly-relevant) features should have negligible contributions.

    References:
        1) Chalasani, Prasad, et al. "Concise explanations of neural networks using adversarial training."
        International Conference on Machine Learning. PMLR, 2020.

    Assumptions:
        Based on authors' implementation as found on the following link:
        https://github.com/jfc43/advex/blob/master/DNN-Experiments/Fashion-MNIST/utils.py.

    """

    @asserts.attributes_check
    def __init__(
        self,
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
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        default_plot_func (callable): Callable that plots the metrics result.
        softmax (boolean): Indicates whether to use softmax probabilities or logits in model prediction, default=False.
        return_aggregate (boolean): Indicates if an aggregated score should be computed over all instances.
        aggregate_func (callable): Callable that aggregates the scores given an evaluation call.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if not abs:
            warn_func.warn_absolute_operation()

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

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "normalising 'normalise' (and 'normalise_func') and if taking absolute"
                    " values of attributions 'abs'"
                ),
                citation=(
                    "Chalasani, Prasad, et al. Concise explanations of neural networks using "
                    "adversarial training.' International Conference on Machine Learning. PMLR, "
                    "(2020)"
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
        softmax: bool = False,
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
        a = np.array(
            np.reshape(a, (np.prod(x.shape[1:]),)),
            dtype=np.float64,
        )
        a += 0.0000001
        a = np.sort(a)
        score = (np.sum((2 * np.arange(1, a.shape[0] + 1) - a.shape[0] - 1) * a)) / (
            a.shape[0] * np.sum(a)
        )
        return score
