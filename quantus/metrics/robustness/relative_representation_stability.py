from __future__ import annotations

from typing import Optional, Callable, Dict, List, Union, TYPE_CHECKING, Tuple
import numpy as np
from functools import partial
import warnings

if TYPE_CHECKING:
    import tensorflow as tf
    import torch
    from quantus import ModelInterface

from quantus.metrics.base_batched import BatchedPerturbationMetric
from quantus.helpers.warn import warn_parameterisation
from quantus.helpers.asserts import attributes_check
from quantus.functions.normalise_func import normalise_by_negative
from quantus.functions.perturb_func import random_noise


class RelativeRepresentationStability(BatchedPerturbationMetric):
    """
    Relative Output Stability leverages the stability of an explanation with respect
    to the change in the output logits

    :math:`RRS(x, x', ex, ex') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{L_x - L_{x'}}{L_x}||_p, \epsilon_{min})},`

    where `L(·)` denotes the internal model representation, e.g., output embeddings of hidden layers.

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/pdf/2203.06877.pdf
    """

    @attributes_check
    def __init__(
        self,
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
        layer_names: Optional[Tuple] = None,
        layer_indices: Optional[Tuple] = None,
        **kwargs: Dict[str, ...],
    ):
        """
        Parameters
        ----------
        nr_samples: int
            The number of samples iterated, default=200.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution.
        normalise: boolean
            Flag stating if the attributions should be normalised
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used, default=gaussian_noise.
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        default_plot_func: callable
            Callable that plots the metrics result.
        eps_min: float
            Small constant to prevent division by 0 in relative_stability_objective, default 1e-6.
        layer_names: tuple, optional
            Names of layers, representations of which should be used for RRS computation, default = all.
        layer_indices: tuple, optional
            Indices of layers, representations of which should be used for RRS computation, default = all.
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
        if layer_names is not None and layer_indices is not None:
            raise ValueError(
                "Must provide either layer_names OR layer_indices, not both."
            )

        self._layer_names = layer_names
        self._layer_indices = layer_indices

        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "function used to generate perturbations 'perturb_func' and parameters passed to it 'perturb_func_kwargs'"
                    "number of times perturbations are sampled 'nr_samples'"
                    "choice which internal representations to use 'layer_names', 'layer_indices'"
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
        **kwargs: Dict[str, ...],
    ) -> Union[List[float], float]:
        """
        Parameters
        ----------
        model: tf.keras.Model, torch.nn.Module
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            4D tensor representing batch of input images
        y_batch: np.ndarray
            1D tensor, representing predicted labels for the x_batch.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        explain_func: callable, optional
            Function used to generate explanations.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        a_batch: np.ndarray, optional
            4D tensor with pre-computed explanations for the x_batch.
        device: str, optional
            Device on which torch should perform computations.
        softmax: boolean, optional
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        kwargs:
            not used, deprecated
        Returns
        -------
        relative representation stability: float, np.ndarray
            float in case `return_aggregate=True`, otherwise np.ndarray of floats


        For each image `x`:
         - generate `num_perturbations` perturbed `xs` in the neighborhood of `x`
         - find `xs` which results in the same label
         - Compute explanations `e_x` and `e_xs`
         - Compute relative representation stability objective, find max value with respect to `xs`
         - In practise we just use `max` over a finite `xs_batch`
        """
        return super(BatchedPerturbationMetric, self).__call__(
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

    def relative_representation_stability_objective(
        self,
        l_x: np.ndarray,
        l_xs: np.ndarray,
        e_x: np.ndarray,
        e_xs: np.ndarray,
    ) -> np.ndarray:
        """
        Computes relative representation stabilities maximization objective
        as defined here https://arxiv.org/pdf/2203.06877.pdf by the authors.

        Parameters
        ----------
        l_x: np.ndarray
            Internal representation for x_batch.
        l_xs: np.ndarray
            Internal representation for xs_batch.
        e_x: np.ndarray
            Explanations for x.
        e_xs: np.ndarray
            Explanations for xs.

        Returns
        -------
        rrs_obj: np.ndarray
            RRS maximization objective.
        """
        # fmt: off
        nominator = (e_x - e_xs) / (e_x + (e_x == 0) * self._eps_min)  # prevent division by 0
        nominator = np.linalg.norm(np.linalg.norm(nominator, axis=(-1, -2)), axis=-1) # noqa
        # fmt: on
        denominator = l_x - l_xs
        denominator /= l_x + (l_x == 0) * self._eps_min  # prevent division by 0

        denominator = np.linalg.norm(denominator, axis=-1)
        denominator += (denominator == 0) * self._eps_min

        return nominator / denominator

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray],
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        model: tf.keras.Model, torch.nn.Module
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            4D tensor representing batch of input images.
        y_batch: np.ndarray
             1D tensor, representing predicted labels for the x_batch.
        a_batch: np.ndarray, optional
            4D tensor with pre-computed explanations for the x_batch.
        args:
            Unused.
        kwargs:
            Unused.

        Returns
        -------
        ris: np.ndarray
            The batched evaluation results.

        """
        _explain_func = partial(
            self.explain_func, model=model.get_model(), **self.explain_func_kwargs
        )
        _perturb_func = partial(self.perturb_func, **self.perturb_func_kwargs)
        if a_batch is None:
            a_batch = self.generate_normalized_explanations_batch(
                x_batch, y_batch, _explain_func
            )

        ris = []
        for _ in range(self._nr_samples):
            # Perturb input.
            x_perturbed = _perturb_func(x_batch)
            labels = model.predict(x_perturbed).argmax(axis=1)
            same_labels_indexes = np.argwhere(y_batch == labels).reshape(-1)
            if len(same_labels_indexes) == 0:
                warnings.warn("Perturbation changed all labels in a batch")
                continue

            _same_labels = np.take(y_batch, same_labels_indexes, axis=0)
            _x_perturbed_batch = np.take(x_perturbed, same_labels_indexes, axis=0)
            _x_batch = np.take(x_batch, same_labels_indexes, axis=0)
            _a_batch = np.take(a_batch, same_labels_indexes, axis=0)
            _a_perturbed_batch = self.generate_normalized_explanations_batch(
                _x_perturbed_batch, _same_labels, _explain_func
            )
            l_x = model.get_hidden_representations(
                _x_batch, self._layer_names, self._layer_indices
            )
            l_x_perturbed = model.get_hidden_representations(
                _x_perturbed_batch, self._layer_names, self._layer_indices
            )
            ris.append(
                self.relative_representation_stability_objective(
                    l_x=l_x,
                    l_xs=l_x_perturbed,
                    e_x=_a_batch,
                    e_xs=_a_perturbed_batch,
                )
            )

        result = np.max(ris, axis=0)
        if self.return_aggregate:
            result = [self.aggregate_func(result)]
        return result
