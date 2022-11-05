# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from functools import partial

if TYPE_CHECKING:
    from typing import Optional, Callable, Dict, List, Tuple
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
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, ...]] = None,
        perturb_func: Callable = None,
        perturb_func_kwargs: Optional[Dict[str, ...]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable[[np.ndarray], np.ndarray]] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        eps_min: float = 1e-6,
        default_plot_func: Optional[Callable] = None,
        layer_names: Optional[Tuple] = None,
        layer_indices: Optional[Tuple] = None,
        return_nan_when_prediction_changes: bool = True,
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
        return_nan_when_prediction_changes: boolean
            When set to true, the metric will be evaluated to NaN if the prediction changes after the perturbation is applied, default=True.
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
        self._return_nan_when_prediction_changes = return_nan_when_prediction_changes

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
        softmax: bool = False,
        channel_first: bool = True,
        **kwargs: Dict[str, ...],
    ) -> List[float] | float:
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

        nominator = (e_x - e_xs) / (
            e_x + (e_x == 0) * self._eps_min
        )  # prevent division by 0
        nominator = np.linalg.norm(np.linalg.norm(nominator, axis=(-1, -2)), axis=-1)
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

        rrs_batch = np.zeros(shape=[self._nr_samples, x_batch.shape[0]])
        for index in range(self._nr_samples):
            x_perturbed = _perturb_func(x_batch)
            a_batch_perturbed = self.generate_normalized_explanations_batch(
                x_perturbed, y_batch, _explain_func
            )
            internal_representations = model.get_hidden_representations(
                x_batch, self._layer_names, self._layer_indices
            )
            internal_representations_perturbed = model.get_hidden_representations(
                x_perturbed, self._layer_names, self._layer_indices
            )
            rrs = self.relative_representation_stability_objective(
                internal_representations,
                internal_representations_perturbed,
                a_batch,
                a_batch_perturbed,
            )

            rrs_batch[index] = rrs

            if not self._return_nan_when_prediction_changes:
                continue

            changed_prediction_indices = np.argwhere(
                model.predict(x_batch).argmax(axis=-1)
                != model.predict(x_perturbed).argmax(axis=-1)
            ).reshape(-1)

            if len(changed_prediction_indices) == 0:
                continue
            rrs_batch[index, changed_prediction_indices] = np.nan

        result = np.max(rrs_batch, axis=0)
        if self.return_aggregate:
            result = [self.aggregate_func(result)]
        return result
