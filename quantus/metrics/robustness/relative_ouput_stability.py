from __future__ import annotations

from typing import Optional, Callable, Dict, List, Union, TYPE_CHECKING
import numpy as np
import functools

if TYPE_CHECKING:
    import tensorflow as tf
    import torch
    from quantus import ModelInterface

from ..base import PerturbationMetric
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.perturb_func import random_noise
from ...helpers.utils import expand_attribution_channel


class RelativeOutputStability(PerturbationMetric):
    """
    Relative Output Stability leverages the stability of an explanation with respect
    to the change in the output logits

    :math:`ROS(x, x', ex, ex') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||h(x) - h(x')}||_p, \epsilon_{min})},`
    where `h(x)` and `h(x')` are the output logits for `x` and `x'` respectively

    References:
            1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations." https://arxiv.org/pdf/2203.06877.pdf
    """

    @asserts.attributes_check
    def __init__(
        self,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[Callable] = None,
        normalise_func_kwargs: Optional[Dict[str, ...]] = None,
        perturb_func: Callable = None,
        perturb_func_kwargs: Optional[Dict[str, ...]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        eps_min: float = 1e-6,
        default_plot_func: Optional[Callable] = None,
        **kwargs: Dict[str, ...],
    ):
        """
        Parameters:
            nr_samples (integer): The number of samples iterated, default=200.
            abs (boolean): Indicates whether absolute operation is applied on the attribution.

            normalise (boolean): a flag stating if the attributions should be normalised
            normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.

            perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=gaussian_noise.
            perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.

            return_aggregate (boolean): Indicates if an aggregated score should be computed over all instances.
            aggregate_func (callable): Callable that aggregates the scores given an evaluation call.

            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            default_plot_func (callable): Callable that plots the metrics result.
            eps_min (float): a small constant to prevent division by 0 in relative_stability_objective, default 1e-6.
        """

        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = random_noise

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

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
        self._nr_samples = nr_samples
        self._eps_min = eps_min

        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "function used to generate perturbations 'perturb_func' and parameters passed to it 'perturb_func_kwargs'"
                    "number of times perturbations are sampled 'nr_samples'"
                ),
                citation='Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations." https://arxiv.org/pdf/2203.06877.pdf',
            )

    def __call__(
        self,
        model: tf.keras.Model | torch.nn.Module,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        model_predict_kwargs: Optional[Dict[str, ...]] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict[str, ...]] = None,
        a_batch: Optional[np.ndarray] = None,
        device: Optional[str] = None,
        softmax: Optional[bool] = False,
        channel_first: Optional[bool] = True,
        **kwargs,
    ) -> Union[List[float], float]:
        """
        Args:
            model: instance of tf.keras.Model or torch.nn.Module
            x_batch: a 4D tensor representing batch of input images
            y_batch: a 1D tensor, representing labels for x_batch. Can be none, if `xs_batch`, `a_batch` and `as_batch` were provided.
            a_batch: a 4D tensor with pre-computed explanations for the x_batch
            explain_func: a function used to generate explanations, must be provided unless a_batch, as_batch were not provided
            device: a device on which torch should perform computations
            softmax: Indicates whether to use softmax probabilities or logits in model prediction. This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
            kwargs: not used, deprecated

        Returns:
            relative output stability: float in case `return_aggregate=True`, otherwise np.ndarray of floats

        For each image `x`:
         - generate `num_perturbations` perturbed `xs` in the neighborhood of `x`
         - find `xs` which results in the same label
         - (or use pre-computed)
         - Compute (or use pre-computed) explanations `e_x` and `e_xs`
         - Compute relative output stability objective, find max value with regard to `xs`
         - In practise we just use `max` over a finite `xs_batch`

        """
        return super(PerturbationMetric, self).__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            a_batch=a_batch,
            device=device,
            softmax=softmax,
            channel_first=channel_first,
            model_predict_kwargs=model_predict_kwargs,
            s_batch=None,
        )

    def relative_output_stability_objective(
        self,
        h_x: np.ndarray,
        h_xs: np.ndarray,
        e_x: np.ndarray,
        e_xs: np.ndarray,
    ) -> np.ndarray:
        """
        Computes relative output stabilities maximization objective
        as defined here https://arxiv.org/pdf/2203.06877.pdf by the authors

        Parameters:
           h_x:  logits for x
           h_xs: logits for xs
           e_x:  explanations for x
           e_xs: explanations for xs
        Returns:
             ros_obj: np.ndarray of float
        """

        nominator = (e_x - e_xs) / (
            e_x + (e_x == 0) * self._eps_min
        )  # prevent division by 0
        nominator = np.linalg.norm(
            np.linalg.norm(nominator, axis=(-1, -2)), axis=-1
        )  # noqa

        denominator = h_x - h_xs

        denominator = np.linalg.norm(denominator, axis=-1)
        denominator += (denominator == 0) * self._eps_min  # prevent division by 0

        return nominator / denominator

    # fmt: off
    def evaluate_instance(self, model: ModelInterface, x: np.ndarray, y: int, a: Optional[np.ndarray] = None, **kwargs) -> float:  # noqa
        # fmt: on
        """
        Args:
            model: model use to generate predictions, explanations
            x: a 3D tensor representing single image
            y: a label for x
            a: pre-computed explanation for x and y
            **kwargs: not used, deprecated

        Returns:
            relative output stability: float
        """
        _explain_func = functools.partial(
            self.explain_func, model=model.get_model(), **self.explain_func_kwargs
        )
        _perturb_func = functools.partial(self.perturb_func, indices=np.arange(0, x.size),
                                          indexed_axes=np.arange(0, x.ndim), **self.perturb_func_kwargs)

        if a is None:
            a = _explain_func(inputs=np.expand_dims(x, 0), targets=np.expand_dims(y, 0))

        x_perturbed_batch = []

        for _ in range(self._nr_samples):
            # Perturb input.
            x_perturbed = _perturb_func(x)
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
            x_perturbed_batch.append(x_perturbed)

        x_perturbed_batch = np.asarray(x_perturbed_batch)
        y_perturbed_batch = model.predict(x_perturbed_batch)

        perturbed_labels = np.argmax(y_perturbed_batch, axis=1)
        same_label_indexes = np.argwhere(perturbed_labels == y).reshape(-1)

        if len(same_label_indexes) == 0:
            raise ValueError("No perturbations resolved in the same labels")

        # only use the perturbations which resolve in the same labels
        x_perturbed_batch = np.take(x_perturbed_batch, same_label_indexes, axis=0)

        # Generate explanation based on perturbed input x.
        a_perturbed_batch = _explain_func(
            inputs=x_perturbed_batch,
            targets=np.full(shape=same_label_indexes.shape, fill_value=y),
        )

        a_perturbed_batch = expand_attribution_channel(a_perturbed_batch, x_perturbed_batch)

        if self.normalise:
            a_perturbed_batch = self.normalise_func(
                a_perturbed_batch, **self.normalise_func_kwargs
            )

        if self.abs:
            a_perturbed_batch = np.abs(a_perturbed_batch)

        h_x = model.predict(np.expand_dims(x, 0))[0]
        h_xs_batch = model.predict(x_perturbed_batch)

        ros_objective_batch = self.relative_output_stability_objective(h_x=h_x, h_xs=h_xs_batch, e_x=a, e_xs=a_perturbed_batch)
        return float(np.max(ros_objective_batch))
