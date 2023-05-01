"""This module contains the implementation of the Avg-Sensitivity metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from quantus.functions.norm_func import fro_norm
from quantus.functions.normalise_func import normalise_by_max
from quantus.functions.perturb_func import uniform_noise
from quantus.functions.similarity_func import difference
from quantus.helpers import asserts, warn, collection_utils, utils, nlp_utils
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.model.text_classifier import Tokenizable
from quantus.helpers.q_types import (
    ExplainFn,
    NormaliseFn,
    PerturbFn,
    AggregateFn,
    SimilarityFn,
    NormFn,
    DataDomain,
)
from quantus.metrics.base_batched import BatchedPerturbationMetric

CLS_DOCSTRING = r"""
    Implementation of {name} by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuring how explanations
    change under slight perturbation - the average sensitivity is captured.

    References:
        1) Chih-Kuan Yeh et al. "On the (in) fidelity and sensitivity for explanations."
        NeurIPS (2019): 10965-10976.
        2) Umang Bhatt et al.: "Evaluating and aggregating
        feature-based model explanations."  IJCAI (2020): 3016-3022.
    """

INIT_DOCSTRING = r"""
    ddd

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
    """

CALL_DOCSTRING = r"""
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


class Sensitivity(BatchedPerturbationMetric):
    data_domain_applicability: List[
        DataDomain
    ] = BatchedPerturbationMetric.data_domain_applicability + ["NLP"]

    @utils.add_docstrings(INIT_DOCSTRING)
    def __init__(
        self,
        similarity_func: SimilarityFn = difference,
        norm_numerator: NormFn = fro_norm,
        norm_denominator: NormFn = fro_norm,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: NormaliseFn = normalise_by_max,
        normalise_func_kwargs: Dict[str, Any] | None = None,
        perturb_func: PerturbFn = uniform_noise,
        perturb_func_kwargs: Dict[str, ...] | None = None,
        return_aggregate: bool = False,
        aggregate_func: AggregateFn = np.mean,
        default_plot_func: Callable[[...], None] | None = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
    ):
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
        )

        self.similarity_func = similarity_func
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator

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

    @utils.add_docstrings(CALL_DOCSTRING)
    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        explain_func: ExplainFn,
        a_batch: np.ndarray | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func_kwargs: dict | None = None,
        model_predict_kwargs: dict | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch: Any | None = None,
        tokenizer: Tokenizable | None = None,
        **kwargs,
    ) -> np.ndarray | float:
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
        self, model, x_batch, x_perturbed, a_batch, a_perturbed, *args, **kwargs
    ):
        if isinstance(x_batch[0], str):
            x_batch, _ = model.get_embeddings(x_batch)

        if isinstance(x_perturbed[0], str):
            x_perturbed, _ = model.get_embeddings(x_perturbed)

        a_batch = nlp_utils.get_scores(a_batch)
        a_perturbed = nlp_utils.get_scores(a_perturbed)

        sensitivities = self.similarity_func(
            utils.flatten_over_axis(a_batch, (0, 1)),
            utils.flatten_over_axis(a_perturbed, (0, 1)),
        )
        numerator = self.norm_numerator(sensitivities)
        denominator = self.norm_denominator(utils.flatten_over_batch(x_batch))
        return numerator / denominator

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray | None,
        a_batch: np.ndarray | None,
        s_batch: np.ndarray,
        custom_batch: np.ndarray | None,
    ) -> None:
        # Additional explain_func assert, as the one in prepare() won't be
        # executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)


@utils.add_docstrings(CLS_DOCSTRING.format(name="Max-Sensitivity"))
class MaxSensitivity(Sensitivity):
    @utils.add_docstrings(INIT_DOCSTRING)
    def __init__(
        self,
        similarity_func: SimilarityFn = difference,
        norm_numerator: NormFn = fro_norm,
        norm_denominator: NormFn = fro_norm,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: NormaliseFn = normalise_by_max,
        normalise_func_kwargs: Dict[str, Any] | None = None,
        perturb_func: PerturbFn = uniform_noise,
        perturb_func_kwargs: Dict[str, ...] | None = None,
        return_aggregate: bool = False,
        aggregate_func: AggregateFn = np.mean,
        default_plot_func: Callable[[...], None] | None = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
    ):
        super().__init__(
            similarity_func=similarity_func,
            norm_numerator=norm_numerator,
            norm_denominator=norm_denominator,
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
        )

    @utils.add_docstrings(CALL_DOCSTRING)
    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        explain_func: ExplainFn,
        a_batch: np.ndarray | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func_kwargs: dict | None = None,
        model_predict_kwargs: dict | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch: Any | None = None,
        tokenizer: Tokenizable | None = None,
        **kwargs,
    ) -> np.ndarray | float:
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

    def reduce_samples(self, similarities: np.ndarray) -> np.ndarray:
        max_func = np.max if self.return_nan_when_prediction_changes else np.nanmax
        return max_func(similarities, axis=0)


@utils.add_docstrings(CLS_DOCSTRING.format(name="Max-Sensitivity"))
class AvgSensitivity(Sensitivity):
    def __init__(
        self,
        similarity_func: SimilarityFn = difference,
        norm_numerator: NormFn = fro_norm,
        norm_denominator: NormFn = fro_norm,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: NormaliseFn = normalise_by_max,
        normalise_func_kwargs: Dict[str, Any] | None = None,
        perturb_func: PerturbFn = uniform_noise,
        perturb_func_kwargs: Dict[str, ...] | None = None,
        return_aggregate: bool = False,
        aggregate_func: AggregateFn = np.mean,
        default_plot_func: Callable[[...], None] | None = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
    ):
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
            similarity_func=similarity_func,
            norm_denominator=norm_denominator,
            norm_numerator=norm_numerator,
        )

    @utils.add_docstrings(CALL_DOCSTRING)
    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        explain_func: ExplainFn,
        a_batch: np.ndarray | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func_kwargs: dict | None = None,
        model_predict_kwargs: dict | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch: Any | None = None,
        tokenizer: Tokenizable | None = None,
        **kwargs,
    ) -> np.ndarray | float:
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

    def reduce_samples(self, similarities: np.ndarray) -> np.ndarray:
        mean_func = np.mean if self.return_nan_when_prediction_changes else np.nanmean
        return mean_func(similarities, axis=0)


class AvgAndMaxSensitivity(Sensitivity):
    """
    Computes average and maximal sensitivity with computational cost of one metric.
    During large scale evaluation, it is often required to compute both.
    Using them separately, means recomputing twice all perturbed explanations,
    and for advanced XAI methods, this is THE MOST computationally expensive part.
    """

    @utils.add_docstrings(INIT_DOCSTRING)
    def __init__(
        self,
        similarity_func: SimilarityFn = difference,
        norm_numerator: NormFn = fro_norm,
        norm_denominator: NormFn = fro_norm,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: NormaliseFn = normalise_by_max,
        normalise_func_kwargs: Dict[str, Any] | None = None,
        perturb_func: PerturbFn = uniform_noise,
        perturb_func_kwargs: Dict[str, ...] | None = None,
        return_aggregate: bool = False,
        aggregate_func: AggregateFn = np.mean,
        default_plot_func: Callable[[...], None] | None = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
    ):
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
            similarity_func=similarity_func,
            norm_denominator=norm_denominator,
            norm_numerator=norm_numerator,
        )

    @utils.add_docstrings(CALL_DOCSTRING)
    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        explain_func: ExplainFn,
        a_batch: np.ndarray | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func_kwargs: dict | None = None,
        model_predict_kwargs: dict | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch: Any | None = None,
        tokenizer: Tokenizable | None = None,
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

    def reduce_samples(self, similarities: np.ndarray) -> dict[str, np.ndarray]:
        mean_func = np.mean if self.return_nan_when_prediction_changes else np.nanmean
        max_func = np.max if self.return_nan_when_prediction_changes else np.nanmax
        avg = mean_func(similarities, axis=0)
        _max = max_func(similarities, axis=0)
        return dict(avg=avg, max=_max)

    @staticmethod
    def join_batches(
        score_batches: List[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        return collection_utils.join_dict(score_batches)
