"""This module contains the implementation of the Random Logit metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import numpy as np
from quantus.helpers.types import (
    SimilarityFn,
    NormaliseFn,
    ExplainFn,
    DataDomain,
    Explanation,
)
from quantus.helpers import asserts
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.functions.normalise_func import normalise_by_max
from quantus.functions.similarity_func import ssim
from quantus.metrics.base_batched import BatchedMetric
from quantus.helpers.nlp_utils import get_scores


class RandomLogit(BatchedMetric):
    """
    Implementation of the Random Logit Metric by Sixt et al., 2020.

    The Random Logit Metric computes the distance between the original explanation and a reference explanation of
    a randomly chosen non-target class.

    References:
        1) Leon Sixt et al.: "When Explanations Lie: Why Many Modified BP
        Attributions Fail." ICML (2020): 9046-9057.
    """

    data_domain_applicability: List[
        DataDomain
    ] = BatchedMetric.data_domain_applicability + ["NLP"]

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: SimilarityFn = ssim,
        num_classes: int = 1000,
        seed: int = 42,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: NormaliseFn = normalise_by_max,
        normalise_func_kwargs: Dict[str, ...] | None = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        default_plot_func: Callable[[...], None] | None = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func: callable
            Similarity function applied to compare input and perturbed input,
            default=ssim.
        num_classes: integer
            Number of prediction classes in the input, default=1000.
        seed: integer
            Seed used for the random generator, default=42.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
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
        kwargs: optional
            Keyword arguments.
        """

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

        # Save metric-specific attributes.
        self.similarity_func = similarity_func
        self.num_classes = num_classes
        self.seed = seed

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("similarity metric 'similarity_func'"),
                citation=(
                    "Sixt, Leon, Granz, Maximilian, and Landgraf, Tim. 'When Explanations Lie: "
                    "Why Many Modified BP Attributions Fail.' arXiv preprint, "
                    "arXiv:1912.09818v6 (2020)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: np.ndarray | List[Explanation] | None = None,
        s_batch: np.ndarray | None = None,
        channel_first: bool | None = None,
        explain_func: ExplainFn = None,
        explain_func_kwargs: Dict[str, ...] | None = None,
        model_predict_kwargs: Dict[str, ...] | None = None,
        softmax: bool | None = False,
        device: str | None = None,
        batch_size: int = 64,
        custom_batch: Any | None = None,
        **kwargs,
    ):
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
            **kwargs,
        )

    def evaluate_batch(
        self,
        model: ModelInterface | TextClassifier,
        x_batch: np.ndarray | List[str],
        y_batch: np.ndarray,
        a_batch: np.ndarray | List[Explanation],
        s_batch: np.ndarray = None,
        custom_batch=None,
    ) -> np.ndarray | float:
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
            Unused.

        Returns
        -------
           : np.ndarray
            The batched evaluation results.
        """
        # Randomly select off-class labels.
        np.random.seed(self.seed)

        def off_label_choice(y):
            return np.random.choice(
                [y_ for y_ in list(np.arange(0, self.num_classes)) if y_ != y]
            )

        y_off = np.asarray([off_label_choice(y) for y in y_batch])
        # Explain against a random class.
        a_off = self.explain_batch(model, x_batch, y_off)

        if isinstance(model, TextClassifier):
            # In case explanation have token in them, drop them.
            a_batch = get_scores(a_batch)
            a_off = get_scores(a_off)

        batch_size = len(x_batch)
        a_batch = np.reshape(a_batch, (batch_size, -1))
        a_off = np.reshape(a_off, (batch_size, -1))

        return self.similarity_func(a_batch, a_off)

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray | None,
        a_batch: np.ndarray | None,
        s_batch: np.ndarray,
        custom_batch: np.ndarray | None,
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        custom_batch: any
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        None
        """
        # Additional explain_func assert, as the one in general_preprocess()
        # won't be executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)
