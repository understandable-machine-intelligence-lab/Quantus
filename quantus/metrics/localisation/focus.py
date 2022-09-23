"""This module contains the implementation of the AUC metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base import Metric
from ...helpers import plotting
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative


class Focus(Metric):
    """
    Implementation of Focus evaluation strategy by Arias et. al. 2022

    The Focus is computed through mosaics of instances from different classes, and the explanations these generate.
    Each mosaic contains four images: two images belonging to the target class (the specific class the feature
    attribution method is expected to explain) and the other two are chosen randomly from the rest of classes.
    Thus, the Focus estimates the reliability of feature attribution method’s output as the probability of the sampled
    pixels lying on an image of the target class of the mosaic. This is equivalent to the proportion
    of positive relevance lying on those images.

    References:
        1) Anna Arias-Duart, Ferran Parés, Dario Garcia-Gasulla, Victor Gimenez-Abalos. "Focus! Rating XAI Methods
        and Finding Biases" arXiv preprint arXiv:2109.15035 (2022).
    """

    @asserts.attributes_check
    def __init__(
        self,
        mosaic_shape: Optional = None,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        default_plot_func: Optional[Callable] = plotting.plot_focus,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
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

        # Save metric-specific attributes.
        self.mosaic_shape = mosaic_shape

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
                    "no parameter! No parameters means nothing to be sensitive on."
                    "Note, however, that Focus only works with image data and "
                    "a 'p_batch' must be provided when calling the metric to "
                    "represent the positions of the target class"
                ),
                citation=(
                    "Arias-Duart, Anna, et al. 'Focus! Rating XAI Methods and Finding Biases.'"
                    "arXiv:2109.15035 (2022)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray],
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
        """
        For this metric to run we need to get the positions of the target class within the mosaic.
        This should be a np.ndarray containing one tuple per sample, representing the positions
        of the target class within the mosaic (where each tuple contains 0/1 values referring to
        (top_left, top_right, bottom_left, bottom_right).

        An example:
            >> custom_batch=[(1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 1, 0), (0, 1, 0, 1)]

        How to initialise the metric and evaluate explanations by calling the metric instance?
            >> metric = Focus()
            >> scores = {method: metric(**init_params)(model=model,
                           x_batch=x_mosaic_batch,
                           y_batch=y_mosaic_batch,
                           a_batch=None,
                           custom_batch=p_mosaic_batch,
                           **{"explain_func": explain,
                              "explain_func_kwargs": {
                              "method": "GradCAM",
                              "gc_layer": "model._modules.get('conv_2')",
                              "pos_only": True,
                              "interpolate": (2*28, 2*28),
                              "interpolate_mode": "bilinear",}
                              "device": device}) for method in ["GradCAM", "IntegratedGradients"]}

            # Plot example!
            >> metric.plot(results=scores)

        """
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

        # Prepare shapes for mosaics.
        self.mosaic_shape = a.shape

        total_positive_relevance = np.sum(a[a > 0], dtype=np.float64)
        target_positive_relevance = 0

        quadrant_functions_list = [
            self.quadrant_top_left,
            self.quadrant_top_right,
            self.quadrant_bottom_left,
            self.quadrant_bottom_right,
        ]

        for quadrant_p, quadrant_func in zip(c, quadrant_functions_list):
            if not bool(quadrant_p):
                continue
            quadrant_relevance = quadrant_func(a=a)
            target_positive_relevance += np.sum(
                quadrant_relevance[quadrant_relevance > 0]
            )

        focus_score = target_positive_relevance / total_positive_relevance

        return focus_score

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
        try:
            assert model is not None
            assert x_batch is not None
            assert y_batch is not None
        except AssertionError:
            raise ValueError(
                "Focus requires either a_batch (explanation maps) or "
                "the necessary arguments to compute it for you (model, x_batch & y_batch)."
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

    def quadrant_top_left(self, a: np.ndarray) -> np.ndarray:
        quandrant_a = a[
            :, : int(self.mosaic_shape[1] / 2), : int(self.mosaic_shape[2] / 2)
        ]
        return quandrant_a

    def quadrant_top_right(self, a: np.ndarray) -> np.ndarray:
        quandrant_a = a[
            :, int(self.mosaic_shape[1] / 2) :, : int(self.mosaic_shape[2] / 2)
        ]
        return quandrant_a

    def quadrant_bottom_left(self, a: np.ndarray) -> np.ndarray:
        quandrant_a = a[
            :, : int(self.mosaic_shape[1] / 2), int(self.mosaic_shape[2] / 2) :
        ]
        return quandrant_a

    def quadrant_bottom_right(self, a: np.ndarray) -> np.ndarray:
        quandrant_a = a[
            :, int(self.mosaic_shape[1] / 2) :, int(self.mosaic_shape[2] / 2) :
        ]
        return quandrant_a
