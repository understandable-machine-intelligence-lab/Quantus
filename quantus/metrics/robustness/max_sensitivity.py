"""This module contains the implementation of the Max-Sensitivity metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
from __future__ import annotations

from functools import singledispatchmethod
from typing import Any, Callable, List, Optional, Dict, TYPE_CHECKING

import numpy as np

from quantus.functions.norm_func import fro_norm
from quantus.functions.normalise_func import normalise_by_max
from quantus.functions.perturb_func import uniform_noise
from quantus.functions.similarity_func import difference
from quantus.helpers import asserts
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.model.text_classifier import TextClassifier, Tokenizable
from quantus.helpers.types import (
    PerturbFn,
    ExplainFn,
    SimilarityFn,
    NormaliseFn,
    NormFn,
    Explanation,
    DataDomain,
    AggregateFn,
)
from quantus.helpers.collection_utils import value_or_default, add_default_items
from quantus.helpers.nlp_utils import (
    is_plain_text_perturbation,
    get_scores,
)
from quantus.metrics.base_batched import BatchedPerturbationMetric

if TYPE_CHECKING:
    import tensorflow as tf
    import torch


class MaxSensitivity(BatchedPerturbationMetric):
    """
    Implementation of Max-Sensitivity by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuring how explanations
    change under slight perturbation - the maximum sensitivity is captured.

    References:
        1) Chih-Kuan Yeh et al. "On the (in) fidelity and sensitivity for explanations."
        NeurIPS (2019): 10965-10976.
        2) Umang Bhatt et al.: "Evaluating and aggregating
        feature-based model explanations."  IJCAI (2020): 3016-3022.
    """

    data_domain_applicability: List[
        DataDomain
    ] = BatchedPerturbationMetric.data_domain_applicability + ["NLP"]

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: SimilarityFn = difference,
        norm_numerator: NormFn = fro_norm,
        norm_denominator: NormFn = fro_norm,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: NormaliseFn = normalise_by_max,
        normalise_func_kwargs: Dict[str, ...] | None = None,
        perturb_func: PerturbFn = uniform_noise,
        lower_bound: float = 0.2,
        upper_bound: float | None = None,
        perturb_func_kwargs: Dict[str, ...] | None = None,
        return_aggregate: bool = False,
        aggregate_func: AggregateFn = np.mean,
        default_plot_func: Callable[[...], None] | None = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func: callable
            Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=difference.
        norm_numerator: callable
            Function for norm calculations on the numerator.
            If None, the default value is used, default=fro_norm
        norm_denominator: callable
            Function for norm calculations on the denominator.
            If None, the default value is used, default=fro_norm
        nr_samples: integer
            The number of samples iterated, default=200.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=gaussian_noise.
        lower_bound: float
            The lower bound of the noise.
        upper_bound: float, optional
            The upper bound of the noise.
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        return_nan_when_prediction_changes: boolean
            When set to true, the metric will be evaluated to NaN if the prediction changes after the perturbation is applied.
        kwargs: optional
            Keyword arguments.
        """
        perturb_func_kwargs = value_or_default(perturb_func_kwargs, lambda: {})
        perturb_func_kwargs = add_default_items(
            perturb_func_kwargs, dict(lower_bound=lower_bound, upper_bound=upper_bound)
        )

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
            nr_samples=nr_samples,
            return_nan_when_prediction_changes=return_nan_when_prediction_changes,
            **kwargs,
        )

        self.similarity_func = value_or_default(similarity_func, lambda: difference)
        self.norm_numerator = value_or_default(norm_numerator, lambda: fro_norm)
        self.norm_denominator = value_or_default(norm_denominator, lambda: fro_norm)

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'lower_bound' and 'upper_bound', the number of samples "
                    "iterated over 'nr_samples', the function to perturb the input "
                    "'perturb_func', the similarity metric 'similarity_func' as well as "
                    "norm calculations on the numerator and denominator of the sensitivity"
                    " equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Yeh, Chih-Kuan, et al. 'On the (in) fidelity and sensitivity for explanations"
                    ".' arXiv preprint arXiv:1901.09392 (2019)"
                ),
            )
            warn.warn_noise_zero(noise=lower_bound)

        self.similarity_func = similarity_func
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator

    def __call__(
        self,
        model: torch.nn.Module | tf.keras.Model | ModelInterface | TextClassifier,
        x_batch: np.array | List[str],
        y_batch: np.array,
        a_batch: Optional[np.ndarray | List[Explanation]] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: ExplainFn = None,
        explain_func_kwargs: Optional[Dict[str, ...]] = None,
        model_predict_kwargs: Optional[Dict[str, ...]] = None,
        softmax: Optional[bool] = False,
        device: Optional[str | torch.device] = None,
        batch_size: int = 64,
        custom_batch: Optional[Any] = None,
        tokenizer: Optional[Tokenizable] = None,
        **kwargs,
    ) -> np.ndarray | float:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        last_results: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """

        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            batch_size=batch_size,
            tokenizer=tokenizer,
            **kwargs,
        )

    @singledispatchmethod
    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: np.ndarray,
        custom_batch=None,
    ) -> np.ndarray:
        """
        Evaluates model and attributes on a single data batch and returns the batched evaluation result.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on an instance-basis.
        y_batch: np.ndarray
            The output to be evaluated on an instance-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s_batch: np.ndarray
            The segmentation to be evaluated on an instance-basis.
        custom_batch:
            Used to pass padded perturbed inputs for plain-text perturbation for NLP.


        Returns
        -------
           : np.ndarray
            The batched evaluation results.
        """
        batch_size = x_batch.shape[0]
        similarities = np.zeros((batch_size, self.nr_samples)) * np.nan

        for step_id in range(self.nr_samples):
            # Perturb input.
            x_perturbed = self.perturb_batch(x_batch)
            changed_prediction_indices = self.changed_prediction_indices(
                model, x_batch, x_perturbed
            )

            for x_instance, x_instance_perturbed in zip(x_batch, x_perturbed):
                warn.warn_perturbation_caused_no_change(
                    x=x_instance,
                    x_perturbed=x_instance_perturbed,
                )
            # Generate explanation based on perturbed input x.
            a_perturbed = self.explain_batch(model, x_perturbed, y_batch)

            # Measure similarity for each instance separately.
            for instance_id in range(batch_size):
                if instance_id in changed_prediction_indices:
                    similarities[instance_id, step_id] = np.nan
                    continue

                sensitivities = self.similarity_func(
                    a=a_batch[instance_id].flatten(),
                    b=a_perturbed[instance_id].flatten(),
                )
                numerator = self.norm_numerator(a=sensitivities)
                denominator = self.norm_denominator(a=x_batch[instance_id].flatten())
                sensitivities_norm = numerator / denominator
                similarities[instance_id, step_id] = sensitivities_norm

        max_func = np.max if self.return_nan_when_prediction_changes else np.nanmax
        return max_func(similarities, axis=1)

    @evaluate_batch.register
    def _(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        s_batch: np.ndarray = None,
        custom_batch=None,
    ) -> np.ndarray:
        batch_size = len(x_batch)
        similarities = np.zeros((batch_size, self.nr_samples)) * np.nan
        is_plain_text = is_plain_text_perturbation(self.perturb_func)

        for step_id in range(self.nr_samples):
            if is_plain_text:
                similarities[:, step_id] = self._eval_step_nlp_plain_text(
                    model, x_batch, y_batch, a_batch, custom_batch[step_id]
                )
            else:
                similarities[:, step_id] = self._eval_step_nlp_embeddings(
                    model, x_batch, y_batch, a_batch
                )

        max_func = np.max if self.return_nan_when_prediction_changes else np.nanmax
        return max_func(similarities, axis=1)

    def _eval_step_nlp_plain_text(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        x_perturbed: List[str],
    ) -> np.ndarray:
        batch_size = len(x_batch)
        similarities = np.zeros(shape=batch_size)
        changed_prediction_indices = self.changed_prediction_indices(
            model, x_batch, x_perturbed
        )

        for x_instance, x_instance_perturbed in zip(x_batch, x_perturbed):
            warn.warn_perturbation_caused_no_change(
                x=x_instance,
                x_perturbed=x_instance_perturbed,
            )
        # Generate explanation based on perturbed input x.
        a_perturbed = self.explain_batch(model, x_perturbed, y_batch)
        # Get numerical part of explanations.
        a_batch = get_scores(a_batch)
        a_perturbed = get_scores(a_perturbed)

        # Get numerical representation of x_batch.
        x_batch_embeddings, _ = model.get_embeddings(x_batch)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if instance_id in changed_prediction_indices:
                similarities[instance_id] = np.nan
                continue

            sensitivities = self.similarity_func(
                a=np.reshape(a_batch[instance_id], -1),
                b=np.reshape(a_perturbed[instance_id], -1),
            )
            numerator = self.norm_numerator(a=sensitivities)
            denominator = self.norm_denominator(
                a=np.reshape(x_batch_embeddings[instance_id], -1)
            )
            sensitivities_norm = numerator / denominator
            similarities[instance_id] = sensitivities_norm

        return similarities

    def _eval_step_nlp_embeddings(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
    ) -> np.ndarray:
        batch_size = len(x_batch)
        similarities = np.zeros(shape=batch_size)

        x_batch_embeddings, predict_kwargs = model.get_embeddings(x_batch)
        a_batch = get_scores(a_batch)

        # Perturb input.
        x_perturbed = self.perturb_batch(x_batch_embeddings)
        changed_prediction_indices = self.changed_prediction_indices(
            model, x_batch, x_perturbed, **predict_kwargs
        )

        for x_instance, x_instance_perturbed in zip(x_batch_embeddings, x_perturbed):
            warn.warn_perturbation_caused_no_change(
                x=x_instance,
                x_perturbed=x_instance_perturbed,
            )
        # Generate explanation based on perturbed input x.
        a_perturbed = self.explain_batch(model, x_perturbed, y_batch)

        # Measure similarity for each instance separately.
        for instance_id in range(batch_size):
            if instance_id in changed_prediction_indices:
                similarities[instance_id] = np.nan
                continue

            sensitivities = self.similarity_func(
                a=np.reshape(a_batch[instance_id], -1),
                b=np.reshape(a_perturbed[instance_id], -1),
            )
            numerator = self.norm_numerator(sensitivities)
            denominator = self.norm_denominator(
                np.reshape(x_batch_embeddings[instance_id], -1)
            )
            sensitivities_norm = numerator / denominator
            similarities[instance_id] = sensitivities_norm

        return similarities
