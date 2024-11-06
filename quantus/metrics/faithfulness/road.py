"""This module contains the implementation of the ROAD metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantus.functions.perturb_func import noisy_linear_imputation
from quantus.helpers import warn
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.perturbation_utils import make_perturb_func
from quantus.metrics.base import Metric

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


@final
class ROAD(Metric[List[float]]):
    """
    Implementation of ROAD evaluation strategy by Rong et al., 2022.

    The ROAD approach measures the accuracy of the model on the provided test set at each step of an iterative process
    of removing k most important pixels. At each step k most relevant pixels (MoRF order) are replaced with noisy linear
    imputations which removes bias.

    Assumptions:
        - The original metric definition relies on perturbation functionality suited only for images.
        Therefore, only apply the metric to 3-dimensional (image) data. To extend the applicablity
        to other data domains, adjustments to the current implementation might be necessary.

    References:
        1) Leemann Rong et al.: "Evaluating Feature Attribution: An Information-Theoretic Perspective." arXiv preprint
        arXiv:2202.00449 (2022).

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "ROAD"
    data_applicability = {DataType.IMAGE}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.LOWER
    evaluation_category = EvaluationCategory.FAITHFULNESS

    def __init__(
        self,
        percentages: Optional[List[float]] = None,
        noise: float = 0.01,
        return_only_values: Optional[bool] = None,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Optional[Callable] = None,
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        percentages: list of ints
            The list of percentages of the image to be removed, default=list(range(1, 100, 2)).
        noise: float
            Noise added, default=0.01.
        return_only_values: bool
            Indicates whether only evaluation scores (list of floats) should be returned and not the dictionary that also includes the percentages, default=None.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
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
        if percentages is None:
            percentages = list(range(1, 100, 2))

        if perturb_func is None:
            perturb_func = noisy_linear_imputation

        self.percentages = percentages
        self.noise = noise
        self.return_values = return_only_values
        self.a_size = None
        self.perturb_func = make_perturb_func(
            perturb_func, perturb_func_kwargs, noise=self.noise
        )

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params="perturbation function 'perturb_func'",
                data_domain_applicability=(
                    f"Also, the current implementation only works for 3-dimensional (image) data."
                ),
                citation=(
                    "Rong, Leemann, et al. 'Evaluating Feature Attribution: An Information-Theoretic Perspective.' "
                    "arXiv:2202.00449 (2022)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

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
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >>> import quantus
            >>> from quantus import LeNet
            >>> import torch

            # Enable GPU.
            >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >>> model = LeNet()
            >>> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >>> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >>> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >>> x_batch, y_batch = iter(test_loader).next()
            >>> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >>> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >>> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >>> metric = Metric(abs=True, normalise=False)
            >>> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
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
            **kwargs,
        )

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
    ) -> List[float]:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.

        Returns
        -------
        list:
            The evaluation results.
        """
        # Order indices.
        ordered_indices = np.argsort(a, axis=None)[::-1]

        results_instance = np.array([None for _ in self.percentages])

        for p_ix, p in enumerate(self.percentages):
            top_k_indices = ordered_indices[: int(self.a_size * p / 100)]

            x_perturbed = self.perturb_func(  # type: ignore
                arr=x,
                indices=top_k_indices,
            )

            warn.warn_perturbation_caused_no_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x and store the difference from predicting on unperturbed input.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            class_pred_perturb = np.argmax(model.predict(x_input))

            # Write a boolean into the percentage results.
            results_instance[p_ix] = int(y == class_pred_perturb)

        # Return list of booleans for each percentage.
        return results_instance

    def custom_batch_preprocess(self, a_batch: np.ndarray, **kwargs) -> None:
        """ROAD requires `a_size` property to be set to `image_height` * `image_width` of an explanation."""
        if self.a_size is None:
            self.a_size = a_batch[0, :, :].size

    def custom_postprocess(
        self,
        **kwargs,
    ) -> None:
        """
        Post-process the evaluation results.

        Parameters
        ----------
        kwargs:
            Unused.

        Returns
        -------
        None
        """

        # Calculate accuracy for every number of most important pixels removed.
        self.evaluation_scores = {
            percentage: np.mean(np.array(self.evaluation_scores)[:, p_ix])
            for p_ix, percentage in enumerate(self.percentages)
        }

        # Return only the evaluation scores (and not percentages).
        if self.return_values:
            self.evaluation_scores = list(self.evaluation_scores.values())

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ) -> List[List[float]]:
        """
        This method performs XAI evaluation on a single batch of explanations.
        For more information on the specific logic, we refer the metric’s initialisation docstring.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        kwargs:
            Unused.

        Returns
        -------
        scores_batch:
            The evaluation results.
        """
        return [
            self.evaluate_instance(model=model, x=x, y=y, a=a)
            for x, y, a in zip(x_batch, y_batch, a_batch)
        ]
