"""This module contains the collection of faithfulness metrics to evaluate attribution-based explanations of neural network models."""
import itertools
import math
import random
import warnings
from multiprocessing import Queue, Semaphore, Manager
from multiprocessing.managers import SyncManager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .base import Metric, BatchedMetric
from .base_parallel import PerturbationMetric, PerturbationMetricSharedIn, PerturbationMetricSharedInOut, PerturbationMetricSharedInReturnOut, PerturbationMetricPassInReturnOut, PerturbationMetricMultiThreadingQueue, PerturbationMetricMultiThreadingPassInReturnOut
from ..helpers import asserts
from ..helpers import plotting
from ..helpers import utils
from ..helpers import perturb_func as perturb_funcs
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative
from ..helpers.similar_func import correlation_pearson, correlation_spearman
from ..helpers.perturb_func import baseline_replacement_by_indices
from ..helpers.perturb_func import baseline_replacement_by_patch


class PixelFlippingMultiProcess(PerturbationMetric):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "black",
            order: str = 'morf',
            features_in_step: int = 1,
            max_steps_per_input: Optional[int] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        if plot_func is None:
            plot_func = plotting.plot_pixel_flipping_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        warn_parametrisation_kwargs = {
            'metric_name': self.__class__.__name__,
            'sensitive_params': ("baseline value 'perturb_baseline'"),
            'citation': (
                "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                "e0130140"
            ),
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            warn_parametrisation_kwargs=warn_parametrisation_kwargs,
            **kwargs,
        )

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.order = order.lower()

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            n_workers: int = 8,
            queue_size: int = 100,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        # TODO: this is incorrect if having multiple channels and perturbing on complete channel
        n_steps = math.ceil(x_batch[0].size / self.features_in_step)
        
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            batch_size=batch_size,
            n_workers=n_workers,
            queue_size=queue_size,
            **kwargs,
        )

    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
            model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=X.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )

        return X, Y, A, S, model

    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            queue,
            n_steps,
            perturb_func,
            perturb_func_kwargs,
            **kwargs,
    ) -> None:
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a_instance.flatten())

        x_perturbed = x_instance.copy().flatten()
        for step in range(n_steps):
            # Perturb input by indices of attributions.
            perturb_start_ix = self.features_in_step * step
            perturb_end_ix = self.features_in_step * (step + 1)
            perturb_indices = a_indices[perturb_start_ix:perturb_end_ix]

            x_perturbed = perturb_func(
                arr=x_perturbed,
                indices=perturb_indices,
                **perturb_func_kwargs,
            )
            #asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
            
            queue.put(
                {
                    'index': index,
                    'step': step,
                    'x': np.reshape(x_perturbed, x_instance.shape),
                    'y': y_instance,
                    'a': a_instance,
                }
            )

    def process_batch(self, model, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        # Predict on perturbed input x.
        x_batch_input = model.shape_input(
            x=x_batch,
            shape=x_batch.shape,
            channel_first=True,
            batched=True,
        )

        y_batch_pred = model.predict(
            x=x_batch_input,
            softmax_act=True,
            #**model_predict_kwargs,
        )

        y_batch_pred_target = y_batch_pred[np.arange(len(y_batch)), y_batch.astype(int).flatten()]
        #print(y_batch_pred_target)
        #breakpoint()
        self.last_results[indices_batch, steps_batch] = y_batch_pred_target
        #self.last_results[indices_batch, steps_batch] = y_batch_pred_target


class PixelFlippingMultiProcessSharedIn(PerturbationMetricSharedIn):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "black",
            order: str = 'morf',
            features_in_step: int = 1,
            max_steps_per_input: Optional[int] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        if plot_func is None:
            plot_func = plotting.plot_pixel_flipping_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        warn_parametrisation_kwargs = {
            'metric_name': self.__class__.__name__,
            'sensitive_params': ("baseline value 'perturb_baseline'"),
            'citation': (
                "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                "e0130140"
            ),
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            warn_parametrisation_kwargs=warn_parametrisation_kwargs,
            **kwargs,
        )

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.order = order.lower()

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            n_workers: int = 8,
            queue_size: int = 100,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        # TODO: this is incorrect if having multiple channels and perturbing on complete channel
        n_steps = math.ceil(x_batch[0].size / self.features_in_step)
        
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            batch_size=batch_size,
            n_workers=n_workers,
            queue_size=queue_size,
            **kwargs,
        )

    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
            model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=X.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )

        return X, Y, A, S, model

    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            queue,
            n_steps,
            perturb_func,
            perturb_func_kwargs,
            **kwargs,
    ) -> None:
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a_instance.flatten())

        x_perturbed = x_instance.copy().flatten()
        for step in range(n_steps):
            # Perturb input by indices of attributions.
            perturb_start_ix = self.features_in_step * step
            perturb_end_ix = self.features_in_step * (step + 1)
            perturb_indices = a_indices[perturb_start_ix:perturb_end_ix]

            x_perturbed = perturb_func(
                arr=x_perturbed,
                indices=perturb_indices,
                **perturb_func_kwargs,
            )
            #asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
            
            queue.put(
                {
                    'index': index,
                    'step': step,
                    'x': np.reshape(x_perturbed, x_instance.shape),
                    'y': y_instance,
                    'a': a_instance,
                }
            )

    def process_batch(self, model, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        # Predict on perturbed input x.
        x_batch_input = model.shape_input(
            x=x_batch,
            shape=x_batch.shape,
            channel_first=True,
            batched=True,
        )

        y_batch_pred = model.predict(
            x=x_batch_input,
            softmax_act=True,
            #**model_predict_kwargs,
        )

        y_batch_pred_target = y_batch_pred[np.arange(len(y_batch)), y_batch.astype(int).flatten()]
        #print(y_batch_pred_target)
        #breakpoint()
        self.last_results[indices_batch, steps_batch] = y_batch_pred_target
        #self.last_results[indices_batch, steps_batch] = y_batch_pred_target


class PixelFlippingMultiProcessSharedInOut(PerturbationMetricSharedInOut):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "black",
            order: str = 'morf',
            features_in_step: int = 1,
            max_steps_per_input: Optional[int] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        if plot_func is None:
            plot_func = plotting.plot_pixel_flipping_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        warn_parametrisation_kwargs = {
            'metric_name': self.__class__.__name__,
            'sensitive_params': ("baseline value 'perturb_baseline'"),
            'citation': (
                "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                "e0130140"
            ),
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            warn_parametrisation_kwargs=warn_parametrisation_kwargs,
            **kwargs,
        )

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.order = order.lower()

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            n_workers: int = 8,
            buffer_size: int = 128,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        # TODO: this is incorrect if having multiple channels and perturbing on complete channel
        n_steps = math.ceil(x_batch[0].size / self.features_in_step)

        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            batch_size=batch_size,
            n_workers=n_workers,
            buffer_size=buffer_size,
            **kwargs,
        )

    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
            model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=X.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )

        return X, Y, A, S, model

    def perturb_and_yield_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            n_steps,
            perturb_func,
            perturb_func_kwargs,
            **kwargs,
    ) -> None:
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a_instance.flatten())

        x_perturbed = x_instance.copy().flatten()
        for step in range(n_steps):
            # Perturb input by indices of attributions.
            perturb_start_ix = self.features_in_step * step
            perturb_end_ix = self.features_in_step * (step + 1)
            perturb_indices = a_indices[perturb_start_ix:perturb_end_ix]

            x_perturbed = perturb_func(
                arr=x_perturbed,
                indices=perturb_indices,
                **perturb_func_kwargs,
            )
            #asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
            
            yield {
                'index': index,
                'step': step,
                'x_perturbed': np.reshape(x_perturbed, x_instance.shape),
            }

    def process_batch(self, model, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        # Predict on perturbed input x.
        x_batch_input = model.shape_input(
            x=x_batch,
            shape=x_batch.shape,
            channel_first=True,
            batched=True,
        )

        y_batch_pred = model.predict(
            x=x_batch_input,
            softmax_act=True,
            #**model_predict_kwargs,
        )

        y_batch_pred_target = y_batch_pred[np.arange(len(y_batch)), y_batch.astype(int).flatten()]
        #print(y_batch_pred_target)
        #breakpoint()
        self.last_results[indices_batch, steps_batch] = y_batch_pred_target
        #self.last_results[indices_batch, steps_batch] = y_batch_pred_target


class PixelFlippingMultiProcessSharedInReturnOut(PerturbationMetricSharedInReturnOut):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "black",
            order: str = 'morf',
            features_in_step: int = 1,
            max_steps_per_input: Optional[int] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        if plot_func is None:
            plot_func = plotting.plot_pixel_flipping_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        warn_parametrisation_kwargs = {
            'metric_name': self.__class__.__name__,
            'sensitive_params': ("baseline value 'perturb_baseline'"),
            'citation': (
                "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                "e0130140"
            ),
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            warn_parametrisation_kwargs=warn_parametrisation_kwargs,
            **kwargs,
        )

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.order = order.lower()

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            n_workers: int = 8,
            buffer_size: int = 128,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        # TODO: this is incorrect if having multiple channels and perturbing on complete channel
        n_steps = math.ceil(x_batch[0].size / self.features_in_step)

        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            batch_size=batch_size,
            n_workers=n_workers,
            buffer_size=buffer_size,
            **kwargs,
        )

    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
            model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=X.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )

        return X, Y, A, S, model

    def perturb_and_yield_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            n_steps,
            perturb_func,
            perturb_func_kwargs,
            **kwargs,
    ) -> None:
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a_instance.flatten())

        x_perturbed = x_instance.copy().flatten()
        for step in range(n_steps):
            # Perturb input by indices of attributions.
            perturb_start_ix = self.features_in_step * step
            perturb_end_ix = self.features_in_step * (step + 1)
            perturb_indices = a_indices[perturb_start_ix:perturb_end_ix]

            x_perturbed = perturb_func(
                arr=x_perturbed,
                indices=perturb_indices,
                **perturb_func_kwargs,
            )
            #asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
            
            yield np.reshape(x_perturbed, x_instance.shape)
            #yield {
            #    'index': index,
            #    'step': step,
            #    'x_perturbed': np.reshape(x_perturbed, x_instance.shape),
            #}

    def process_batch(self, model, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        # Predict on perturbed input x.
        x_batch_input = model.shape_input(
            x=x_batch,
            shape=x_batch.shape,
            channel_first=True,
            batched=True,
        )

        y_batch_pred = model.predict(
            x=x_batch_input,
            softmax_act=True,
            #**model_predict_kwargs,
        )

        y_batch_pred_target = y_batch_pred[np.arange(len(y_batch)), y_batch.astype(int).flatten()]
        #print(y_batch_pred_target)
        #breakpoint()
        self.last_results[indices_batch, steps_batch] = y_batch_pred_target
        #self.last_results[indices_batch, steps_batch] = y_batch_pred_target


class PixelFlippingMultiProcessPassInReturnOut(PerturbationMetricPassInReturnOut):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "black",
            order: str = 'morf',
            features_in_step: int = 1,
            max_steps_per_input: Optional[int] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        if plot_func is None:
            plot_func = plotting.plot_pixel_flipping_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        warn_parametrisation_kwargs = {
            'metric_name': self.__class__.__name__,
            'sensitive_params': ("baseline value 'perturb_baseline'"),
            'citation': (
                "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                "e0130140"
            ),
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            warn_parametrisation_kwargs=warn_parametrisation_kwargs,
            **kwargs,
        )

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.order = order.lower()

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            n_workers: int = 8,
            buffer_size: int = 128,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        # TODO: this is incorrect if having multiple channels and perturbing on complete channel
        n_steps = math.ceil(x_batch[0].size / self.features_in_step)

        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            batch_size=batch_size,
            n_workers=n_workers,
            buffer_size=buffer_size,
            **kwargs,
        )

    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
            model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=X.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )

        return X, Y, A, S, model

    def perturb_and_yield_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            n_steps,
            perturb_func,
            perturb_func_kwargs,
            **kwargs,
    ) -> None:
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a_instance.flatten())

        x_perturbed = x_instance.copy().flatten()
        for step in range(n_steps):
            # Perturb input by indices of attributions.
            perturb_start_ix = self.features_in_step * step
            perturb_end_ix = self.features_in_step * (step + 1)
            perturb_indices = a_indices[perturb_start_ix:perturb_end_ix]

            x_perturbed = perturb_func(
                arr=x_perturbed,
                indices=perturb_indices,
                **perturb_func_kwargs,
            )
            #asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
            
            yield np.reshape(x_perturbed, x_instance.shape)
            #yield {
            #    'index': index,
            #    'step': step,
            #    'x_perturbed': np.reshape(x_perturbed, x_instance.shape),
            #}

    def process_batch(self, model, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        # Predict on perturbed input x.
        x_batch_input = model.shape_input(
            x=x_batch,
            shape=x_batch.shape,
            channel_first=True,
            batched=True,
        )

        y_batch_pred = model.predict(
            x=x_batch_input,
            softmax_act=True,
            #**model_predict_kwargs,
        )

        y_batch_pred_target = y_batch_pred[np.arange(len(y_batch)), y_batch.astype(int).flatten()]
        #print(y_batch_pred_target)
        #breakpoint()
        self.last_results[indices_batch, steps_batch] = y_batch_pred_target
        #self.last_results[indices_batch, steps_batch] = y_batch_pred_target


class PixelFlippingMultiThreadingQueue(PerturbationMetricMultiThreadingQueue):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "black",
            order: str = 'morf',
            features_in_step: int = 1,
            max_steps_per_input: Optional[int] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        if plot_func is None:
            plot_func = plotting.plot_pixel_flipping_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        warn_parametrisation_kwargs = {
            'metric_name': self.__class__.__name__,
            'sensitive_params': ("baseline value 'perturb_baseline'"),
            'citation': (
                "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                "e0130140"
            ),
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            warn_parametrisation_kwargs=warn_parametrisation_kwargs,
            **kwargs,
        )

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.order = order.lower()

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            n_workers: int = 8,
            queue_size: int = 100,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        # TODO: this is incorrect if having multiple channels and perturbing on complete channel
        n_steps = math.ceil(x_batch[0].size / self.features_in_step)
        
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            batch_size=batch_size,
            n_workers=n_workers,
            queue_size=queue_size,
            **kwargs,
        )

    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
            model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=X.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )

        return X, Y, A, S, model

    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            batch_queue,
            n_steps,
            perturb_func,
            perturb_func_kwargs,
            **kwargs,
    ) -> None:
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a_instance.flatten())

        x_perturbed = x_instance.copy().flatten()
        for step in range(n_steps):
            # Perturb input by indices of attributions.
            perturb_start_ix = self.features_in_step * step
            perturb_end_ix = self.features_in_step * (step + 1)
            perturb_indices = a_indices[perturb_start_ix:perturb_end_ix]

            x_perturbed = perturb_func(
                arr=x_perturbed,
                indices=perturb_indices,
                **perturb_func_kwargs,
            )
            #asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
            
            batch_queue.put(
                {
                    'index': index,
                    'step': step,
                    'x': np.reshape(x_perturbed, x_instance.shape),
                    'y': y_instance,
                    'a': a_instance,
                }
            )

    def process_batch(self, model, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        # Predict on perturbed input x.
        x_batch_input = model.shape_input(
            x=x_batch,
            shape=x_batch.shape,
            channel_first=True,
            batched=True,
        )

        y_batch_pred = model.predict(
            x=x_batch_input,
            softmax_act=True,
            #**model_predict_kwargs,
        )

        y_batch_pred_target = y_batch_pred[np.arange(len(y_batch)), y_batch.astype(int).flatten()]
        #print(y_batch_pred_target)
        #breakpoint()
        self.last_results[indices_batch, steps_batch] = y_batch_pred_target
        #self.last_results[indices_batch, steps_batch] = y_batch_pred_target


class PixelFlippingMultiThreadingPassInReturnOut(PerturbationMetricMultiThreadingPassInReturnOut):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "black",
            order: str = 'morf',
            features_in_step: int = 1,
            max_steps_per_input: Optional[int] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        if plot_func is None:
            plot_func = plotting.plot_pixel_flipping_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        warn_parametrisation_kwargs = {
            'metric_name': self.__class__.__name__,
            'sensitive_params': ("baseline value 'perturb_baseline'"),
            'citation': (
                "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                "e0130140"
            ),
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            warn_parametrisation_kwargs=warn_parametrisation_kwargs,
            **kwargs,
        )

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.order = order.lower()

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            n_workers: int = 8,
            buffer_size: int = 128,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        # TODO: this is incorrect if having multiple channels and perturbing on complete channel
        n_steps = math.ceil(x_batch[0].size / self.features_in_step)

        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            batch_size=batch_size,
            n_workers=n_workers,
            buffer_size=buffer_size,
            **kwargs,
        )

    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
            model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=X.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )

        return X, Y, A, S, model

    def perturb_and_yield_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            n_steps,
            perturb_func,
            perturb_func_kwargs,
            **kwargs,
    ) -> None:
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a_instance.flatten())

        x_perturbed = x_instance.copy().flatten()
        for step in range(n_steps):
            # Perturb input by indices of attributions.
            perturb_start_ix = self.features_in_step * step
            perturb_end_ix = self.features_in_step * (step + 1)
            perturb_indices = a_indices[perturb_start_ix:perturb_end_ix]

            x_perturbed = perturb_func(
                arr=x_perturbed,
                indices=perturb_indices,
                **perturb_func_kwargs,
            )
            #asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
            
            yield np.reshape(x_perturbed, x_instance.shape)
            #yield {
            #    'index': index,
            #    'step': step,
            #    'x_perturbed': np.reshape(x_perturbed, x_instance.shape),
            #}

    def process_batch(self, model, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        # Predict on perturbed input x.
        x_batch_input = model.shape_input(
            x=x_batch,
            shape=x_batch.shape,
            channel_first=True,
            batched=True,
        )

        y_batch_pred = model.predict(
            x=x_batch_input,
            softmax_act=True,
            #**model_predict_kwargs,
        )

        y_batch_pred_target = y_batch_pred[np.arange(len(y_batch)), y_batch.astype(int).flatten()]
        #print(y_batch_pred_target)
        #breakpoint()
        self.last_results[indices_batch, steps_batch] = y_batch_pred_target
        #self.last_results[indices_batch, steps_batch] = y_batch_pred_target

