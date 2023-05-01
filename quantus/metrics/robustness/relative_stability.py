# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional, Callable, Dict

import numpy as np

if TYPE_CHECKING:
    pass

from quantus.helpers import collection_utils
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.base_batched import BatchedPerturbationMetric
from quantus.helpers.warn import warn_parameterisation
from quantus.functions.normalise_func import normalise_by_average_second_moment_estimate
from quantus.functions.perturb_func import uniform_noise
from quantus.functions.norm_func import l2_norm

from quantus.helpers.utils import get_logits_for_labels
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.collection_utils import value_or_default, safe_as_array
from quantus.helpers.nlp_utils import get_scores

INIT_DOCSTRING = """
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
        return_nan_when_prediction_changes: boolean
            When set to true, the metric will be evaluated to NaN if the prediction changes after the perturbation is applied, default=True.
        """

CALL_DOCSTRING = """
        For each image `x`:
         - Generate `num_perturbations` perturbed `xs` in the neighborhood of `x`.
         - Compute explanations `e_x` and `e_xs`.
         - Compute relative input stability objective, find max value with respect to `xs`.
         - In practise we just use `max` over a finite `xs_batch`.

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
        batch_size: int
            The batch size to be used.
        kwargs:
            not used, deprecated
        Returns
        -------
        relative input stability: float, np.ndarray
            float in case `return_aggregate=True`, otherwise np.ndarray of floats
        """


class RelativeInputStability(BatchedPerturbationMetric):
    """
    Relative Input Stability leverages the stability of an explanation with respect to the change in the input data

    :math:`RIS(x, x', e_x, e_x') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{x - x'}{x}||_p, \epsilon_{min})}`

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/abs/2203.06877
    """

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
        aggregate_func: Optional[Callable[[np.ndarray], np.float]] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        default_plot_func: Optional[Callable] = None,
        return_nan_when_prediction_changes: bool = True,
        **kwargs,
    ):
        normalise_func = value_or_default(
            normalise_func, lambda: normalise_by_average_second_moment_estimate
        )
        perturb_func = value_or_default(perturb_func, lambda: uniform_noise)
        perturb_func_kwargs = value_or_default(
            perturb_func_kwargs, lambda: {"upper_bound": 0.2}
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

        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "function used to generate perturbations 'perturb_func' and parameters passed to it 'perturb_func_kwargs'"
                    "number of times perturbations are sampled 'nr_samples'"
                ),
                citation='Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations." https://arxiv.org/pdf/2203.06877.pdf',
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        explain_func,
        a_batch: np.ndarray | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func_kwargs: dict | None = None,
        model_predict_kwargs: dict | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch=None,
        tokenizer=None,
        **kwargs,
    ) -> dict[str, np.ndarray | float]:
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

    def evaluate_sample(
        self,
        model: TextClassifier | ModelInterface,
        x_batch,
        x_perturbed,
        a_batch,
        a_perturbed,
        *args,
        **kwargs,
    ):
        if isinstance(x_batch[0], str):
            x_batch, _ = model.get_embeddings(x_batch)

        if isinstance(x_perturbed[0], str):
            x_perturbed, _ = model.get_embeddings(x_perturbed)

        return RelativeInputStability.relative_input_stability_objective(
            x_batch, x_perturbed, get_scores(a_batch), get_scores(a_perturbed)
        )

    @staticmethod
    def relative_input_stability_objective(
        x: np.ndarray,
        xs: np.ndarray,
        e_x: np.ndarray,
        e_xs: np.ndarray,
    ) -> np.ndarray:
        """
        Computes relative input stability's maximization objective
         as defined here :ref:`https://arxiv.org/pdf/2203.06877.pdf` by the authors.
        Parameters
        ----------
        x: np.ndarray
            Batch of images.
        xs: np.ndarray
            Batch of perturbed images.
        e_x: np.ndarray
            Explanations for x.
        e_xs: np.ndarray
            Explanations for xs.

        Returns
        -------
        ris_obj: np.ndarray
            RIS maximization objective.
        """

        # prevent division by 0
        eps_min = np.finfo(np.float32).eps

        nominator = (e_x - e_xs) / (e_x + eps_min)
        nominator = l2_norm(nominator)

        denominator = x - xs
        denominator /= x + eps_min
        denominator = l2_norm(denominator) + eps_min

        return nominator / denominator

    def reduce_samples(self, scores):
        return np.max(scores, axis=0)


class RelativeOutputStability(BatchedPerturbationMetric):
    """
    Relative Output Stability leverages the stability of an explanation with respect
    to the change in the output logits

    :math:`ROS(x, x', ex, ex') = max \\frac{||\\frac{e_x - e_x'}{e_x}||_p}{max (||h(x) - h(x')||_p, \epsilon_{min})}`,

    where `h(x)` and `h(x')` are the output logits for `x` and `x'` respectively


    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/pdf/2203.06877.pdf
    """

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        explain_func,
        a_batch: np.ndarray | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func_kwargs: dict | None = None,
        model_predict_kwargs: dict | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch=None,
        tokenizer=None,
        **kwargs,
    ) -> dict[str, np.ndarray | float]:
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
        aggregate_func: Optional[Callable[[np.ndarray], np.float]] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        default_plot_func: Optional[Callable] = None,
        return_nan_when_prediction_changes: bool = True,
        **kwargs,
    ):
        normalise_func = value_or_default(
            normalise_func, lambda: normalise_by_average_second_moment_estimate
        )
        perturb_func = value_or_default(perturb_func, lambda: uniform_noise)
        perturb_func_kwargs = value_or_default(
            perturb_func_kwargs, lambda: {"upper_bound": 0.2}
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

        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "function used to generate perturbations 'perturb_func' and parameters passed to it 'perturb_func_kwargs'"
                    "number of times perturbations are sampled 'nr_samples'"
                ),
                citation='Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations." https://arxiv.org/pdf/2203.06877.pdf',
            )

    def evaluate_sample(
        self, model, x_batch, x_perturbed, a_batch, a_perturbed, y_batch, predict_kwargs
    ):
        h_x = get_logits_for_labels(
            safe_as_array(model.predict(x_batch, **predict_kwargs), force=True), y_batch
        )
        h_xs = get_logits_for_labels(
            safe_as_array(model.predict(x_perturbed, **predict_kwargs), force=True),
            y_batch,
        )

        return RelativeOutputStability.relative_output_stability_objective(
            h_x, h_xs, get_scores(a_batch), get_scores(a_perturbed)
        )

    @staticmethod
    def relative_output_stability_objective(
        h_x: np.ndarray,
        h_xs: np.ndarray,
        e_x: np.ndarray,
        e_xs: np.ndarray,
    ) -> np.ndarray:
        """
        Computes relative output stability's maximization objective
         as defined here :ref:`https://arxiv.org/pdf/2203.06877.pdf` by the authors.

        Parameters
        ----------
        h_x: np.ndarray
            Output logits for x_batch.
        h_xs: np.ndarray
            Output logits for xs_batch.
        e_x: np.ndarray
            Explanations for x.
        e_xs: np.ndarray
            Explanations for xs.

        Returns
        -------
        ros_obj: np.ndarray
            ROS maximization objective.
        """

        # prevent division by 0
        eps_min = np.finfo(np.float32).eps

        nominator = (e_x - e_xs) / (e_x + eps_min)
        nominator = l2_norm(nominator)

        denominator = h_x - h_xs
        denominator = l2_norm(denominator) + eps_min
        return nominator / denominator

    def reduce_samples(self, scores):
        return np.max(scores, axis=0)


class RelativeRepresentationStability(BatchedPerturbationMetric):
    """
    Relative Output Stability leverages the stability of an explanation with respect
    to the change in the output logits

    :math:`RRS(x, x', ex, ex') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{L_x - L_{x'}}{L_x}||_p, \epsilon_{min})},`

    where `L(·)` denotes the internal model representation, e.g., output embeddings of hidden layers.

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/pdf/2203.06877.pdf
    """

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
        aggregate_func: Optional[Callable[[np.ndarray], np.float]] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        default_plot_func: Optional[Callable] = None,
        return_nan_when_prediction_changes: bool = True,
        **kwargs,
    ):
        normalise_func = value_or_default(
            normalise_func, lambda: normalise_by_average_second_moment_estimate
        )
        perturb_func = value_or_default(perturb_func, lambda: uniform_noise)
        perturb_func_kwargs = value_or_default(
            perturb_func_kwargs, lambda: {"upper_bound": 0.2}
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

        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "function used to generate perturbations 'perturb_func' and parameters passed to it 'perturb_func_kwargs'"
                    "number of times perturbations are sampled 'nr_samples'"
                ),
                citation='Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations." https://arxiv.org/pdf/2203.06877.pdf',
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        explain_func,
        a_batch: np.ndarray | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func_kwargs: dict | None = None,
        model_predict_kwargs: dict | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch=None,
        tokenizer=None,
        **kwargs,
    ) -> dict[str, np.ndarray | float]:
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

    def evaluate_sample(
        self, model, x_batch, x_perturbed, a_batch, a_perturbed, y_batch, predict_kwargs
    ):
        l_x = model.get_hidden_representations(x_batch, **predict_kwargs)
        l_xs = model.get_hidden_representations(x_perturbed, **predict_kwargs)

        return (
            RelativeRepresentationStability.relative_representation_stability_objective(
                l_x, l_xs, get_scores(a_batch), get_scores(a_perturbed)
            )
        )

    @staticmethod
    def relative_representation_stability_objective(
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

        # prevent division by 0
        eps_min = np.finfo(np.float32).eps

        nominator = (e_x - e_xs) / (e_x + eps_min)
        nominator = l2_norm(nominator)

        denominator = l_x - l_xs
        denominator /= l_x + eps_min
        denominator = l2_norm(denominator) + eps_min

        return nominator / denominator

    def reduce_samples(self, scores):
        return np.max(scores, axis=0)


class CombinedRelativeStability(BatchedPerturbationMetric):
    """Computes 3 metrics with computational cost of 1."""

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
        aggregate_func: Optional[Callable[[np.ndarray], np.float]] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        default_plot_func: Optional[Callable] = None,
        return_nan_when_prediction_changes: bool = True,
        **kwargs,
    ):
        normalise_func = value_or_default(
            normalise_func, lambda: normalise_by_average_second_moment_estimate
        )
        perturb_func = value_or_default(perturb_func, lambda: uniform_noise)
        perturb_func_kwargs = value_or_default(
            perturb_func_kwargs, lambda: {"upper_bound": 0.2}
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

        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "function used to generate perturbations 'perturb_func' and parameters passed to it 'perturb_func_kwargs'"
                    "number of times perturbations are sampled 'nr_samples'"
                ),
                citation='Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations." https://arxiv.org/pdf/2203.06877.pdf',
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        explain_func,
        a_batch: np.ndarray | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func_kwargs: dict | None = None,
        model_predict_kwargs: dict | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch=None,
        tokenizer=None,
        **kwargs,
    ) -> dict[str, np.ndarray | float]:
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

    def evaluate_sample(
        self, model, x_batch, x_perturbed, a_batch, a_perturbed, y_batch, predict_kwargs
    ):
        h_x = get_logits_for_labels(
            safe_as_array(model.predict(x_batch, **predict_kwargs), force=True), y_batch
        )
        h_xs = get_logits_for_labels(
            safe_as_array(model.predict(x_perturbed, **predict_kwargs), force=True),
            y_batch,
        )

        l_x = model.get_hidden_representations(x_batch, **predict_kwargs)
        l_xs = model.get_hidden_representations(x_perturbed, **predict_kwargs)

        if isinstance(x_batch[0], str):
            x_batch, _ = model.get_embeddings(x_batch)

        if isinstance(x_perturbed[0], str):
            x_perturbed, _ = model.get_embeddings(x_perturbed)

        a_batch = get_scores(a_batch)
        a_perturbed = get_scores(a_perturbed)
        ris = RelativeInputStability.relative_input_stability_objective(
            x_batch, x_perturbed, a_batch, a_perturbed
        )
        ros = RelativeOutputStability.relative_output_stability_objective(
            h_x, h_xs, a_batch, a_perturbed
        )
        rrs = (
            RelativeRepresentationStability.relative_representation_stability_objective(
                l_x, l_xs, a_batch, a_perturbed
            )
        )

        return dict(ris=ris, ros=ros, rrs=rrs)

    def reduce_samples(self, scores):
        scores_combined = collection_utils.join_dict(scores)
        return collection_utils.map_dict(scores_combined, partial(np.max, axis=0))
