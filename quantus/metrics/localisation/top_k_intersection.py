"""This module contains the implementation of the Top-K Intersection metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base import Metric
from ...helpers import asserts
from ...helpers import warn_func
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative


class TopKIntersection(Metric):
    """
    Implementation of the top-k intersection by Theiner et al., 2021.

    The TopKIntersection implements the pixel-wise intersection between a ground truth target object mask and
    an "explainer" mask, the binarized version of the explanation. High scores are desired, as the
    overlap between the ground truth object mask and the attribution mask should be maximal.

    References:
        1) Theiner, Jonas, Müller-Budack Eric, and Ewerth, Ralph. "Interpretable Semantic Photo
        Geolocalization." arXiv preprint arXiv:2104.14995 (2021).
    """

    @asserts.attributes_check
    def __init__(
        self,
        k: int = 1000,
        concept_influence: bool = False,
        abs: bool = False,
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
        k (integer): Top k attributions values to use, default=1000.
        concept_influence (boolean): Indicates whether concept influence metric is used, default=False.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
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
        self.k = k
        self.concept_influence = concept_influence

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch', the number of features to "
                    "consider 'k', if size of the ground truth mask is taking into account"
                    " 'concept_influence' as well as if attributions are normalised "
                    "'normalise' (and 'normalise_func') and/ or taking absolute values "
                    "of such 'abs'"
                ),
                citation=(
                    "Theiner, Jonas, Müller-Budack Eric, and Ewerth, Ralph. 'Interpretable "
                    "Semantic Photo Geolocalization.' arXiv preprint arXiv:2104.14995 (2021)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray],
        s_batch: np.array,
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
    ):

        if np.sum(s) == 0:
            warn_func.warn_empty_segmentation()
            return np.nan

        # Prepare shapes.
        s = s.astype(bool)
        top_k_binary_mask = np.zeros(a.shape)

        # Sort and create masks.
        sorted_indices = np.argsort(a, axis=None)
        np.put_along_axis(top_k_binary_mask, sorted_indices[-self.k :], 1, axis=None)
        top_k_binary_mask = top_k_binary_mask.astype(bool)

        # Top-k intersection.
        tki = 1.0 / self.k * np.sum(np.logical_and(s, top_k_binary_mask))

        # Concept influence (with size of object normalised tki score).
        if self.concept_influence:
            tki = np.prod(s.shape) / np.sum(s) * tki

        return tki

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

        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)
        asserts.assert_value_smaller_than_input_size(
            x=x_batch, value=self.k, value_name="k"
        )
        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
