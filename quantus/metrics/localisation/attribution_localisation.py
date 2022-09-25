"""This module contains the implementation of the Attribution Localisation metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base import Metric
from ...helpers import asserts
from ...helpers import warn_func
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative


class AttributionLocalisation(Metric):
    """
    Implementation of the Attribution Localization by Kohlbrenner et al., 2020.

    The Attribution Localization implements the ratio of positive attributions within the target to the overall
    attribution. High scores are desired, as it means, that the positively attributed pixels belong to the
    targeted object class.

    References:
        1) Kohlbrenner M., Bauer A., Nakajima S., Binder A., Wojciech S., Lapuschkin S.
           "Towards Best Practice in Explaining Neural Network Decisions with LRP."
           arXiv preprint arXiv:1910.09840v2 (2020).

    """

    @asserts.attributes_check
    def __init__(
        self,
        weighted: bool = False,
        max_size: float = 1.0,
        abs: bool = True,
        normalise: bool = True,
        normalise_func: Optional[Callable] = None,
        normalise_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        default_plot_func: Optional[Callable] = None,
        display_progressbar: bool = False,
        disable_warnings: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        weighted (boolean): Indicates whether the weighted variant of the inside-total relevance ratio is used,
            default=False.
        max_size (float): The maximum ratio for  the size of the bounding box to image, default=1.0.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
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

        if not abs:
            warn_func.warn_absolute_operation()

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
        self.weighted = weighted
        self.max_size = max_size

        # Asserts and warnings.
        self.disable_warnings = disable_warnings
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch', if size of the ground truth "
                    "mask is taking into account 'weighted' as well as if attributions"
                    " are normalised 'normalise' (and 'normalise_func') and/ or taking "
                    "the absolute values of such 'abs'"
                ),
                citation=(
                    "Kohlbrenner M., Bauer A., Nakajima S., Binder A., Wojciech S., Lapuschkin S. "
                    "'Towards Best Practice in Explaining Neural Network Decisions with LRP."
                    "arXiv preprint arXiv:1910.09840v2 (2020)."
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
    ) -> float:

        if np.sum(s) == 0:
            warn_func.warn_empty_segmentation()
            return np.nan

        # Prepare shapes.
        a = a.flatten()
        s = s.flatten().astype(bool)

        # Compute ratio.
        size_bbox = float(np.sum(s))
        size_data = np.prod(x.shape[1:])
        ratio = size_bbox / size_data

        # Compute inside/outside ratio.
        inside_attribution = np.sum(a[s])
        total_attribution = np.sum(a)
        inside_attribution_ratio = float(inside_attribution / total_attribution)

        if ratio <= self.max_size:
            if inside_attribution_ratio > 1.0:
                warn_func.warn_segmentation(inside_attribution, total_attribution)
                return np.nan
            if not self.weighted:
                return inside_attribution_ratio
            else:
                return float(inside_attribution_ratio * ratio)

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

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
