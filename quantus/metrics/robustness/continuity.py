"""This module contains the implementation of the Continuity metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import itertools
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantus.functions.perturb_func import batched_translation
from quantus.functions.similarity_func import distance_euclidean, distance_manhattan
from quantus.helpers import asserts, utils, warn
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
class Continuity(Metric[List[float]]):
    """
    Implementation of the Continuity test by Montavon et al., 2018.

    The test measures the strongest variation of the explanation in the input domain i.e.,
    ||R(x) - R(x')||_1 / ||x - x'||_2
    where R(x) is the explanation for input x and x' is the perturbed input.

    Assumptions:
        - The original metric definition relies on perturbation functionality suited only for images.
        Therefore, only apply the metric to 3-dimensional (image) data. To extend the applicablity
        to other data domains, adjustments to the current implementation might be necessary.

    References:
        1) Grégoire Montavon et al.: "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Continuity"
    data_applicability = {DataType.IMAGE}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.LOWER
    evaluation_category = EvaluationCategory.ROBUSTNESS

    def __init__(
        self,
        nr_steps: int = 28,
        abs: bool = True,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Optional[Callable] = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
        **kwargs,
    ):
        """
        Parameters
            ----------
        patch_size: integer
            The patch size for masking, default=7.
        nr_steps: integer
            The number of steps to iterate over, default=28.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=True.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=translation_x_direction.
        perturb_baseline: string
            Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
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
        default_plot_func: callable
            Callable that plots the metrics result.
        return_nan_when_prediction_changes: boolean
            When set to true, the metric will be evaluated to NaN if the prediction changes after the perturbation is applied.
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

        if perturb_func is None:
            perturb_func = batched_translation

        # Save metric-specific attributes.
        self.nr_steps = nr_steps
        self.nr_patches: Optional[int] = None
        self.return_nan_when_prediction_changes = return_nan_when_prediction_changes
        self.perturb_func = make_perturb_func(perturb_func, perturb_func_kwargs, perturb_baseline=perturb_baseline)

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "how many patches to split the input image to 'nr_patches', "
                    "the number of steps to iterate over 'nr_steps', the value to replace"
                    " the masking with 'perturb_baseline' and in what direction to "
                    "translate the image 'perturb_func'"
                ),
                data_domain_applicability=(
                    f"Also, the current implementation only works for 3-dimensional (image) data."
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73, 1-15 (2018"
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
        softmax: Optional[bool] = False,
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
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

    def custom_preprocess(
        self,
        x_batch: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        kwargs:
            Unused.
        Returns
        -------
        None.
        """

        # Asserts.
        # Additional explain_func assert, as the one in prepare() won't be
        # executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        This method performs XAI evaluation on a single batch of explanations.
        For more information on the specific logic, we refer the metric’s initialisation docstring.

        Parameters
        ----------
        model:
            A model that is subject to explanation.
        x_batch:
            A np.ndarray which contains the input data that are explained.
        y_batch:
            A np.ndarray which contains the output labels that are explained.
        kwargs:
            Unused.

        Returns
        -------
        scores_batch:
            Evaluation results.
        """
        # Expand height dimension if not present
        if len(x_batch.shape) == 3:
            x_batch = x_batch[:, :, None, :]
        total_steps_possible = x_batch.shape[3] * 2 + 1
        dx = total_steps_possible // self.nr_steps
        batch_size = x_batch.shape[0]
        current_translation = -x_batch.shape[3]
        x_input = model.shape_input(x_batch, x_batch.shape, channel_first=True, batched=True)

        # Retrieve the original attibutions and preprocess (if needed)
        a_batch = self.explain_batch(model, x_input, y_batch)
        if self.normalise:
            a_batch = self.normalise_func(a_batch)
        if self.abs:
            a_batch = np.abs(a_batch)
        a_batch = a_batch.reshape(batch_size, -1)

        # Translate the image step by step
        results = []
        for step in range(self.nr_steps):
            current_translation += dx
            x_batch_perturbed = batched_translation(x_batch, "black", current_translation, direction="x")
            x_input = model.shape_input(x_batch_perturbed, x_batch.shape, channel_first=True, batched=True)
            y_pred_perturb = model.predict(x_input)

            prediction_changed = y_pred_perturb.argmax(-1) != y_batch
            y_pred_perturb_value = y_pred_perturb[np.arange(batch_size), y_batch]
            a_perturbed = self.explain_batch(model, x_input, y_batch)

            if self.normalise:
                a_perturbed = self.normalise_func(a_perturbed)
            if self.abs:
                a_perturbed = np.abs(a_perturbed)
            a_perturbed = a_perturbed.reshape(batch_size, -1)

            # Calculate continuity, as per the original definition
            result = distance_manhattan(a_batch, a_perturbed) / (
                distance_euclidean(x_batch.reshape(batch_size, -1), x_batch_perturbed.reshape(batch_size, -1)) + 1e-9
            )
            # Set NaN to positions where prediction changed
            if self.return_nan_when_prediction_changes:
                result[prediction_changed] = np.NaN
            # Save result
            results.append(result)
        results = np.stack(results, axis=1)
        results = np.max(results, axis=1)

        return results
