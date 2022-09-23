"""This module contains the implementation of the Iterative Removal of Features metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base import PerturbationMetric
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers import utils
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.perturb_func import baseline_replacement_by_indices


class IROF(PerturbationMetric):
    """
    Implementation of IROF (Iterative Removal of Features) by Rieger at el., 2020.

    The metric computes the area over the curve per class for sorted mean importances
    of feature segments (superpixels) as they are iteratively removed (and prediction scores are collected),
    averaged over several test samples.

    References:
        1) Rieger, Laura, and Lars Kai Hansen. "Irof: a low resource evaluation metric for
        explanation methods." arXiv preprint arXiv:2003.08747 (2020).

    """

    @asserts.attributes_check
    def __init__(
        self,
        segmentation_method: str = "slic",
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        perturb_baseline: str = "mean",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = True,
        aggregate_func: Optional[Callable] = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        segmentation_method (string): Image segmentation method:'slic' or 'felzenszwalb', default="slic".
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="mean".
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate (boolean): Indicates if an aggregated score should be computed over all instances.
        aggregate_func (callable): Callable that aggregates the scores given an evaluation call.
            default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        self.segmentation_method = segmentation_method
        self.nr_channels = None

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the method to segment "
                    "the image 'segmentation_method' (including all its associated"
                    " hyperparameters), also, IROF only works with image data"
                ),
                citation=(
                    "Rieger, Laura, and Lars Kai Hansen. 'Irof: a low resource evaluation metric "
                    "for explanation methods.' arXiv preprint arXiv:2003.08747 (2020)"
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

        # Predict on x.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        # Segment image.
        segments = utils.get_superpixel_segments(
            img=np.moveaxis(x, 0, -1).astype("double"),
            segmentation_method=self.segmentation_method,
        )
        nr_segments = segments.max()
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(a[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        preds = []

        for i_ix, s_ix in enumerate(s_indices):

            # Perturb input by indices of attributions.
            a_ix = np.nonzero((segments == s_ix).flatten())[0]

            x_perturbed = self.perturb_func(
                arr=x,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])

            # Normalise the scores to be within range [0, 1].
            preds.append(float(y_pred_perturb / y_pred))

        # Calculate the area over the curve (AOC) score.
        aoc = len(preds) - utils.calculate_auc(np.array(preds))
        return aoc

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

        # Infer number of input channels.
        self.nr_channels = x_batch.shape[1]

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )

    @property
    def get_aoc_score(self):
        """Calculate the area over the curve (AOC) score for several test samples."""
        return np.mean(self.last_results)
