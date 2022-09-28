from __future__ import annotations

from typing import Optional, Callable, Dict, List, Union, TYPE_CHECKING
import numpy as np
import importlib.util

if TYPE_CHECKING:
    import tensorflow as tf
    import torch

from ... import (
    attributes_check,
    normalise_by_negative,
    PerturbationMetric,
    uniform_noise,
    warn_parameterisation,
    ModelInterface,
    assert_perturbation_caused_change,
    infer_channel_first,
    get_wrapped_model
)







class RelativeInputStability(PerturbationMetric):
    """
    Relative Input Stability leverages the stability of an explanation with respect to the change in the input data

    :math:`RIS(x, x', ex, ex') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{x - x'}{x}||_p, \epsilon_{min})}`

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations." https://arxiv.org/pdf/2203.06877.pdf
    """

    @attributes_check
    def __init__(self,
                 nr_samples: int = 200,
                 abs=False,

                 normalise=False,
                 normalise_func: Optional[Callable] = None,
                 normalise_func_kwargs: Optional[Dict[str, ...]] = None,

                 perturb_func: Callable = None,
                 perturb_func_kwargs: Optional[Dict[str, ...]] = None,

                 return_aggregate=False,
                 aggregate_func: Optional[Callable] = np.mean,

                 disable_warnings=False,
                 display_progressbar=False,

                 eps_min=1e-6,
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
            perturb_func = uniform_noise

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
        self.nr_samples = nr_samples
        self.eps_min = eps_min

        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "function used to generate perturbations 'perturb_func' and parameters passed to it 'perturb_func_kwargs'"
                    "number of times perturbations are sampled 'nr_samples'"
                ),
                citation="Chirag Agarwal, et. al., 2022. \"Rethinking stability for attribution based explanations.\" https://arxiv.org/pdf/2203.06877.pdf"
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
            **kwargs,
    ) -> float | np.ndarray:
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
            ris: float in case `return_aggregate=True`, otherwise np.ndarray of floats

        For each image `x`:
         - generate `num_perturbations` perturbed `xs` in the neighborhood of `x`
         - find `xs` which results in the same label
         - (or use pre-computed)
         - Compute (or use pre-computed) explanations `e_x` and `e_xs`
         - Compute relative input stability objective, find max value with regard to `xs`
         - In practise we just use `max` over a finite `xs_batch`

        """
        if model_predict_kwargs is None:
            model_predict_kwargs = dict()
        self.model_predict_kwargs = model_predict_kwargs

        return super(PerturbationMetric, self).__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            a_batch=a_batch,
            device=device,
            softmax=softmax,
            channel_first=None,
            model_predict_kwargs=self.model_predict_kwargs,
            s_batch=None
        )


    def relative_input_stability_objective(self, x: np.ndarray, xs: np.ndarray, e_x: np.ndarray, e_xs: np.ndarray) -> np.ndarray:
        """
        Computes relative input stabilities maximization objective
        as defined here https://arxiv.org/pdf/2203.06877.pdf by the authors

        Args:
            x:    4D tensor of datapoints with shape (batch_size, ...)
            xs:   4D tensor of perturbed datapoints with shape (batch_size, ...)
            e_x:  4D tensor of explanations for x with shape (batch_size, ...)
            e_xs: 4D tensor of explanations for xs with shape (batch_size, ...)
        Returns:
            ris_obj: A 1D tensor with shape (batch_size,)
        """

        nominator = (e_x - e_xs) / (e_x + (e_x == 0) * self.eps_min)  # prevent division by 0
        if len(nominator.shape) == 3:
            # In practise quantus.explain often returns tensors of shape (batch_size, img_height, img_width)
            nominator = np.linalg.norm(nominator, axis=(2, 1))  # noqa
        else:
            nominator = np.linalg.norm(
                np.linalg.norm(nominator, axis=(3, 2)), axis=1  # noqa
            )  # noqa

        denominator = x - xs
        denominator /= x + (x == 0) * self.eps_min

        denominator = np.linalg.norm(
            np.linalg.norm(denominator, axis=(3, 2)), axis=1  # noqa
        )  # noqa
        denominator += (denominator == 0) * self.eps_min

        return nominator / denominator

    def evaluate_instance(
            self,
            i: int,
            model: ModelInterface,
            x: np.ndarray,
            y: int,
            a: Optional[np.ndarray] = None,
            c: Optional[np.ndarray] = None,
            p: Optional[np.ndarray] = None,
            **kwargs
    ) -> float:
        """
        Args:
            i: ???
            model:
            x: a 3D tensor representing single image
            y: a label for x
            a: pre-computed explanation for x and y
            s: not used
            c:
            p:

        Returns:

        """
        results = []
        for _ in range(self.nr_samples):

            # Perturb input.
            x_perturbed = self.perturb_func(
                arr=x,
                indices=np.arange(0, x.size),
                indexed_axes=np.arange(0, x.ndim),
                **self.perturb_func_kwargs,
            )
            assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Generate explanation based on perturbed input x.
            a_perturbed = self.explain_func(
                model=model.get_model(),
                inputs=x,
                targets=y,
                **self.explain_func_kwargs,
            )

            y_perturbed = model.predict(np.expand_dims(x, 0), **self.model_predict_kwargs)
            if y_perturbed[0] != y

            if self.normalise:
                a_perturbed = self.normalise_func(
                    a_perturbed, **self.normalise_func_kwargs
                )

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            ris_objective = self.relative_input_stability_objective(x, x_perturbed, a, a_perturbed)

            results.append(ris_objective)

        # Append average sensitivity score.
        return float(np.max(ris_objective))
